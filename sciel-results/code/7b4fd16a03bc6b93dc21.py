import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    min_n = 10000
    max_n = 1000000
    limit = max_n + 1
    N = limit + 1
    
    lpf = np.zeros(N, dtype=np.int32)
    
    # Sieve for largest prime factors
    for i in range(2, limit + 1):
        if time.time() > deadline:
            return min_n # Fallback, though unlikely
        if lpf[i] == 0:
            # i is prime
            lpf[i::i] = i
            
    # Compute M(n) = max(lpf[n], lpf[n+1])
    left = lpf[min_n:max_n + 1]
    right = lpf[min_n + 1:max_n + 2]
    
    max_factors = np.maximum(left, right)
    min_idx = np.argmin(max_factors)
    
    best_n = min_n + min_idx
    
    return int(best_n)
