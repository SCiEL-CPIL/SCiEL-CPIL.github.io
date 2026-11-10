import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    size = N_MAX + 2  # To check n+1 up to 1000001
    
    # Precompute largest prime factor for all numbers up to size
    lpf = np.zeros(size, dtype=np.int32)
    for i in range(2, size):
        if lpf[i] == 0:  # i is prime
            for j in range(i, size, i):
                lpf[j] = i
    
    # Now iterate y from 2 upwards
    for y in range(2, 200):
        if time.time() > deadline:
            return N_MIN
        
        # Vectorized check: n is y-smooth if lpf[n] <= y
        valid_n = (lpf[N_MIN:N_MAX+1] <= y) & (lpf[N_MIN+1:N_MAX+2] <= y)
        
        if np.any(valid_n):
            first_idx = np.argmax(valid_n)
            return N_MIN + int(first_idx)
    
    # Fallback
    return N_MIN
