import time
import numpy as np

N = 100
START_TIME = time.time()
DEADLINE = START_TIME + 145  # 5s safety margin before 150s kill
OPTIMAL_STRING = np.ones(N, dtype=int).astype('U1').join('')
