from flask import Flask
from flask import jsonify
from flask import request
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import traceback
from sqlalchemy import create_engine
import math

app = Flask(__name__)


def create_dataset(dataset, look_back=1):  # look_back为滑窗
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def deal(type_, Country, Province):
    path_name = 'datas/time_series_covid19_' + type_ + '_global.csv'
    confirmed_global_df = pd.DataFrame(pd.read_csv(path_name))
    start_date = 0
    end_date = (len(confirmed_global_df.loc[0]) - 4)
    days_in_future = 3
    future_forcast = np.array([i for i in range(end_date - start_date + days_in_future)]).reshape(-1, 1)  # 未来预测序数列

    first_day = datetime.datetime.strptime('1/22/20', '%m/%d/%y')
    future_forcast_dates = []
    future_forcast_dates2 = []
    for i in range(len(future_forcast)):
        future_forcast_dates.append((first_day + datetime.timedelta(days=i + start_date)).strftime('%m/%d'))
        future_forcast_dates2.append((first_day + datetime.timedelta(days=i + start_date)).strftime('%m/%d/%y'))
    adjusted_dates = future_forcast_dates[:-days_in_future]  # 矫正后日期start-end,eg:'1/22','1/23'……
    dates_array = np.array([i for i in range(end_date - start_date)]).reshape(-1, 1)  # 开始到结束日准确序数列[1，2，3，……]
    Country_Province_df = confirmed_global_df.loc[confirmed_global_df["Country/Region"] == Country]  # 对应的省份选项
    if (len(Country_Province_df) == 1):
        confirmed_Province_df = Country_Province_df
        confirmed_Province_array = np.array(confirmed_Province_df.iloc[:, start_date + 4:end_date + 4]).reshape(-1, 1)
    else:
        if Province == "sum":
            confirmed_Province_array = np.array(list(Country_Province_df.sum()[4:])).reshape(-1, 1)
        else:
            confirmed_Province_df = Country_Province_df[Country_Province_df["Province/State"] == Province]
            confirmed_Province_array = np.array(confirmed_Province_df.iloc[:, start_date + 4:end_date + 4]).reshape(-1,
                                                                                                                    1)
    X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(dates_array,
                                                                                                confirmed_Province_array,
                                                                                                test_size=0.1,
                                                                                                shuffle=False)

    # LSTM
    dataset = confirmed_Province_array
    dataset_or = dataset
    np.random.seed(7)
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    train = dataset
    look_back = 7
    trainX, trainY = create_dataset(train, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='max')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    trainPredict = model.predict(trainX)  # 预测训练集
    list_sum = []
    x2 = np.array([trainX[-1]])
    zzz = model.predict(x2)
    list_sum.append(zzz[0])
    list_1 = list(trainX[-1][0])[1:7]
    list_1.append(float(zzz[0]))
    list_1 = np.array([[list_1]])
    zzz = model.predict(list_1)
    list_sum.append(zzz[0])
    list_2 = list(list_1[0][0])[1:7]
    list_2.append(float(zzz[0]))
    list_2 = np.array([[list_2]])
    zzz = model.predict(list_2)
    list_sum.append(zzz[0])
    list_sum = scaler.inverse_transform(list_sum)
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    LSTM_RMSE = math.sqrt(mean_squared_error(trainY[0, :], trainPredict[:, 0]))
    list_sum = [int(i) for i in list_sum]
    list_lstm = [0 for j in range(7)] + [int(i) for i in trainPredict] + list_sum

    # 多项式回归
    poly_reg = PolynomialFeatures(degree=3)
    x_poly = poly_reg.fit_transform(X_train_confirmed)
    linear_reg = LinearRegression()
    linear_reg.fit(x_poly, y_train_confirmed)
    Linear_pred = linear_reg.predict(poly_reg.fit_transform(future_forcast))
    Linear_RMSE = math.sqrt(mean_squared_error(confirmed_Province_array[:, 0], Linear_pred[:-3, 0]))

    # SVM
    svm = SVR()
    svm_grid = {'shrinking': False, 'kernel': 'linear', 'gamma': 0.01, 'epsilon': 1, 'C': 10}
    svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True,
                                    n_jobs=-1, n_iter=30, verbose=1)
    svm.fit(X_train_confirmed, y_train_confirmed)
    Svm_pred = svm.predict(future_forcast)
    Svm_RMSE = math.sqrt(mean_squared_error(confirmed_Province_array[:, 0], Svm_pred[:-3]))
    list_svm = [int(i) for i in list(Svm_pred)]

    # 多元线性回归
    linear_reg2 = LinearRegression()
    linear_reg2.fit(X_train_confirmed, y_train_confirmed)
    list_pred = linear_reg2.predict(future_forcast)
    list_pred2 = [int(i) for i in list_pred]
    Linear_RMSE2 = math.sqrt(mean_squared_error(confirmed_Province_array[:, 0], list_pred[:-3]))

    list_confirmed = sum(confirmed_Province_array.tolist(), [])
    list_linear = [int(i) for i in sum(Linear_pred.tolist(), [])]
    list_days = future_forcast_dates2
    return list_lstm, list_confirmed, list_svm, list_pred2, list_linear, list_days, [LSTM_RMSE, Linear_RMSE,
                                                                                     Linear_RMSE2, Svm_RMSE]


@app.route('/api/getconfired', methods=["GET"])
def getconfired():
    try:
        type_ = request.args.get("type")
        country = request.args.get("country")
        province = request.args.get("province")
        list_lstm, list_confirmed, list_svm, list_linear, list_linear2, list_days, RMSE_data = deal(type_, country,
                                                                                                    province)
        return jsonify(
            {'lstm_data': list_lstm, "confirmed_data": list_confirmed, "svm_data": list_svm, "liner_data": list_linear,
             "linear2_data": list_linear2, "dates": list_days, "RMSE_data": RMSE_data})
    except:
        traceback.print_exc()
        return jsonify({"errs": "there is no data!!!"})


@app.route('/api/daystart', methods=["GET"])
def daystart():
    try:
        confirmed_global_df = pd.DataFrame(pd.read_csv('datas/time_series_covid19_confirmed_global.csv'))
        conn = create_engine('mysql+pymysql://root:520xy1314@localhost:3305/zph_bs?charset=utf8').connect()
        data2 = confirmed_global_df[["Province/State", "Country/Region"]]
        data2.to_sql('country_province', conn, if_exists='replace', index=False)
        conn.close()
        return jsonify({"msg": "success!!!"})
    except:
        return jsonify({"msg": "errs!!!"})


@app.route('/api/daydownload', methods=["GET"])
def daydownload():
    try:
        return jsonify({"msg": "success!!!"})
    except:
        return jsonify({"msg": "errs!!!"})


@app.route('/api/getcountry', methods=["GET"])
def getcountry():
    country = request.args.get("country")
    if country:
        confirmed_df = pd.read_csv('datas/time_series_covid19_confirmed_global.csv')
        deaths_df = pd.read_csv('datas/time_series_covid19_deaths_global.csv')
        recoveries_df = pd.read_csv('datas/time_series_covid19_recovered_global.csv')
        cols = confirmed_df.keys()
        confirmed = list(confirmed_df.loc[confirmed_df["Country/Region"] == country, cols[4]:cols[-1]].sum())
        deaths = list(deaths_df.loc[deaths_df["Country/Region"] == country, cols[4]:cols[-1]].sum())
        recoveries = list(recoveries_df.loc[recoveries_df["Country/Region"] == country, cols[4]:cols[-1]].sum())
        deaths_rate_list = []
        recovered_rate_list = []
        for i in range(len(confirmed)):
            if confirmed[i] == 0:
                deaths_rate = 0
                recovered_rate = 0
            else:
                deaths_rate = round((deaths[i] / confirmed[i]) * 100, 2)
                recovered_rate = round((recoveries[i] / confirmed[i]) * 100, 2)
            deaths_rate_list.append(deaths_rate)
            recovered_rate_list.append(recovered_rate)
        start_date = 0
        end_date = (len(confirmed_df.loc[0]) - 4)
        future_forcast = np.array([i for i in range(end_date - start_date)]).reshape(-1, 1)  # 未来预测序数列

        first_day = datetime.datetime.strptime('1/22/20', '%m/%d/%y')
        future_forcast_dates2 = []
        for i in range(len(future_forcast)):
            future_forcast_dates2.append((first_day + datetime.timedelta(days=i + start_date)).strftime('%Y-%m-%d'))
        return jsonify({"confirmed_data": confirmed, "deaths_data": deaths, "recovered_data": recoveries,
                        "deaths_rate": deaths_rate_list, "recovered_rate": recovered_rate_list,
                        "dates": future_forcast_dates2})
    else:
        return jsonify({"errs": "there is no data!!!"})


@app.route('/api/getmostcountrys', methods=['GET'])
def getmostcountrys():
    confirmed_df = pd.read_csv('datas/time_series_covid19_confirmed_global.csv')
    deaths_df = pd.read_csv('datas/time_series_covid19_deaths_global.csv')
    recoveries_df = pd.read_csv('datas/time_series_covid19_recovered_global.csv')
    dates = list(confirmed_df.keys())[4:]
    list_sum = []
    for country in confirmed_df.sort_values(by=dates[-1])['Country/Region'][-10:]:
        confirmed = confirmed_df.loc[confirmed_df["Country/Region"] == country, dates[-1]].sum()
        deaths = deaths_df.loc[deaths_df["Country/Region"] == country, dates[-1]].sum()
        recovereds = recoveries_df.loc[recoveries_df["Country/Region"] == country, dates[-1]].sum()
        list_sum.append([country, int(confirmed), int(deaths), int(recovereds)])
    list_sum.reverse()
    return jsonify(list_sum)


if __name__ == '__main__':
    app.run()
