import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 78  # safety margin before 90s limit
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_val = 1000024
    
    # Precompute small prime part using sieve-like method
    # For each prime p and each power p^k, multiply small_part[m] by p
    # for all m divisible by p^k
    small_part = np.ones(max_val + 1, dtype=np.int64)
    
    for p in primes:
        p_power = p
        while p_power <= max_val:
            # Vectorized: multiply small_part at indices p_power, 2*p_power, ...
            indices = np.arange(p_power, max_val + 1, p_power)
            small_part[indices] *= p
            p_power *= p
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        return 1000
    
    # Search for n that minimizes distinct count
    best_count = float('inf')
    best_n = 1000
    
    # Use a faster distinct count: for 24 elements, use a set
    for n in range(1000, 1000001):
        if time.time() > deadline:
            break
        
        factors = small_part[n+1:n+25]
        # Convert to tuple for fast hashing
        distinct_count = len(set(factors.tolist()))
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                break
    
    return best_n
