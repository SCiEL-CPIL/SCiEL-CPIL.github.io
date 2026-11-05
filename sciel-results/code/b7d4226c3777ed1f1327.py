import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    N = 1000001
    # Precompute largest prime factor for all numbers up to N
    lpf = np.zeros(N + 1, dtype=np.int32)
    
    # Sieve: for each prime, update all its multiples
    for i in range(2, N + 1):
        if lpf[i] == 0:  # i is prime
            lpf[i::i] = i  # Since we go in increasing order, last assignment is the largest prime factor
    
    # Now scan for the best n using vectorized operations
    # Compute max(lpf[n], lpf[n+1]) for all n in [10000, 999999]
    lpf_arr = lpf[10000:N]  # lpf[10000] to lpf[999999]
    lpf_shifted = lpf[10001:N+1]  # lpf[10001] to lpf[1000000]
    
    max_lpf = np.maximum(lpf_arr, lpf_shifted)
    
    # Find the index of the minimum value
    min_idx = np.argmin(max_lpf)
    best_n = 10000 + min_idx
    best_val = max_lpf[min_idx]
    
    return int(best_n)
