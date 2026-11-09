import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 55  # Safety margin for 60s limit
    
    N = 1000000
    max_val = N + 1  # We need lpf for N+1 as well
    
    # Step 1: Find all primes up to max_val using a simple boolean sieve
    is_prime = np.ones(max_val + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(max_val)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    # Step 2: Compute LPF by marking multiples in ascending prime order
    # (larger primes overwrite, so final value is the largest prime factor)
    lpf = np.zeros(max_val + 1, dtype=np.int32)
    primes = np.where(is_prime)[0]
    for p in primes:
        lpf[p::p] = p
    
    if time.time() - start_time > deadline:
        return 10000
    
    # Step 3: Vectorized scan for minimum max(lpf[n], lpf[n+1])
    n_range = np.arange(10000, N + 1, dtype=np.int32)
    lpf_n = lpf[n_range]
    lpf_np1 = lpf[n_range + 1]
    max_lpf = np.maximum(lpf_n, lpf_np1)
    
    min_idx = np.argmin(max_lpf)
    best_n = int(n_range[min_idx])
    
    return best_n
