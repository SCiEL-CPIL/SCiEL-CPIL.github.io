import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # Precompute largest prime factor for all numbers up to N_MAX+1
    limit = N_MAX + 1
    
    # Sieve to find largest prime factor
    lpf = np.zeros(limit + 1, dtype=np.int64)
    for i in range(2, limit + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, limit + 1, i):
                lpf[j] = i
    
    # Scan for n in [N_MIN, N_MAX] minimizing max(lpf[n], lpf[n+1])
    n_range = np.arange(N_MIN, N_MAX + 1)
    lpf_n = lpf[n_range]
    lpf_np1 = lpf[n_range + 1]
    scores = np.maximum(lpf_n, lpf_np1)
    
    best_idx = np.argmin(scores)
    best_n = int(n_range[best_idx])
    
    return best_n
