import numpy as np
import time

def construct():
    start_time = time.time()
    MAX_N = 1000000
    LIMIT = MAX_N + 1
    DEADLINE = start_time + 55.0  # 5s buffer

    # 1. Sieve of Eratosthenes to find primes up to LIMIT
    is_prime = np.ones(LIMIT + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(np.sqrt(LIMIT)) + 1):
        if is_prime[i]:
            is_prime[i*i : LIMIT+1 : i] = False
            
    primes = np.where(is_prime)[0]
    
    # 2. Compute Largest Prime Factor (LPF)
    lpf = np.arange(LIMIT + 1, dtype=np.int64)
    
    # Iterate through all primes to ensure LPF is correct for all numbers
    for p in primes:
        if time.time() > DEADLINE:
            break
        lpf[p : LIMIT+1 : p] = p
        
    # 3. Find n in [10000, MAX_N] minimizing max(lpf[n], lpf[n+1])
    n_start = 10000
    n_end = MAX_N
    
    vals_n = lpf[n_start : n_end + 1]
    vals_np1 = lpf[n_start + 1 : n_end + 2]
    
    max_lpf = np.maximum(vals_n, vals_np1)
    
    min_val = np.min(max_lpf)
    best_idx = np.argmin(max_lpf)
    best_n = n_start + best_idx
    
    return int(best_n)
