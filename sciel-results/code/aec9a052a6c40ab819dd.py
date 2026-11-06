import math
import time
import numpy as np

def construct():
    N_MIN = 10000
    N_MAX = 1000000
    DEADLINE = 85
    
    start_time = time.time()
    
    max_val = N_MAX + 1
    
    # Sieve for Largest Prime Factor
    # lpf[i] will store the largest prime factor of i
    lpf = [1] * (max_val + 1)
    
    # Sieve of Eratosthenes variant
    for i in range(2, max_val + 1):
        if lpf[i] == 1: # i is prime
            for j in range(i, max_val + 1, i):
                lpf[j] = i
                
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return N_MIN

    # Vectorized scan
    lpf_arr = np.array(lpf, dtype=np.int64)
    a = lpf_arr[N_MIN:N_MAX + 1]
    b = lpf_arr[N_MIN + 1:N_MAX + 2]
    vals = np.maximum(a, b)
    best_idx = int(np.argmin(vals))
    best_n = N_MIN + best_idx
    
    return best_n
