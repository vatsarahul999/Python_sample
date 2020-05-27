import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn import metrics


joblib_file = "joblib_RL_Model.pkl"
joblib_LR_model = joblib.load(joblib_file)

df = pd.read_csv('USA_Housing.csv')
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.4,random_state=101)
predict = joblib_LR_model.predict(X_test)
print(predict)
mean_avg_error = metrics.mean_absolute_error(y_test,predict)
print('mean_avg_error : '+ str(mean_avg_error))