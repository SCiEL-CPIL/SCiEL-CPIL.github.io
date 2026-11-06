import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N_MIN = 10000
    N_MAX = 1000000
    limit = N_MAX + 1
    
    # Sieve of Eratosthenes
    is_prime = np.ones(limit + 1, dtype=np.bool_)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i : limit+1 : i] = False
    
    primes = np.where(is_prime)[0]
    
    # Compute Largest Prime Factor
    lpf = np.zeros(limit + 1, dtype=np.int64)
    for p in primes:
        # Vectorized update
        lpf[p : limit+1 : p] = p
        
    lpf_n = lpf[N_MIN:N_MAX+1]
    lpf_n1 = lpf[N_MIN+1:N_MAX+2]
    
    max_lpf = np.maximum(lpf_n, lpf_n1)
    min_val = np.min(max_lpf)
    min_indices = np.where(max_lpf == min_val)[0]
    best_n = N_MIN + min_indices[0]
    
    return int(best_n)
