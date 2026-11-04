import time
START_TIME = time.time()
TIME_LIMIT = 140.0 # 10s margin

def check_time():
    return time.time() - START_TIME > TIME_LIMIT
