import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    N_MIN = 10000
    N_MAX = 1000000
    limit = N_MAX + 1
    
    # Sieve of Eratosthenes using NumPy for speed
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    # Only need to sieve up to sqrt(limit)
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = False
            
    primes = np.where(is_prime)[0]
    
    # Compute Largest Prime Factor (LPF)
    # lpf[x] is the largest prime factor of x
    lpf = np.zeros(limit + 1, dtype=np.int32)
    
    # Iterate through primes
    for p in primes:
        # For each multiple of p, set lpf to p
        # Since p is increasing, the last assignment is the largest prime factor
        lpf[p:limit+1:p] = p
        
    # Find n in [N_MIN, N_MAX] minimizing max(lpf[n], lpf[n+1])
    # We can vectorize this part too
    # Create array of n values
    # n ranges from N_MIN to N_MAX
    # We need lpf[n] and lpf[n+1]
    
    # To save memory and time, we can just loop, but NumPy slicing is fast
    # However, creating an array of 990,000 elements is fine.
    
    # Let's try a vectorized approach for the search
    # n_arr = np.arange(N_MIN, N_MAX + 1)
    # v1 = lpf[n_arr]
    # v2 = lpf[n_arr + 1]
    # vals = np.maximum(v1, v2)
    # min_idx = np.argmin(vals)
    # best_n = N_MIN + min_idx
    # best_val = vals[min_idx]
    
    # This is O(N) memory and very fast.
    
    if time.time() > deadline - 2:
        # Fallback if sieve took too long (unlikely)
        return N_MIN

    n_arr = np.arange(N_MIN, N_MAX + 1)
    v1 = lpf[n_arr]
    v2 = lpf[n_arr + 1]
    vals = np.maximum(v1, v2)
    
    min_idx = np.argmin(vals)
    best_n = N_MIN + min_idx
    
    return int(best_n)
