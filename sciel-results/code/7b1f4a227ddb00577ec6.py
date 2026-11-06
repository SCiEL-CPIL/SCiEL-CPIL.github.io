# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    MAX_N = 1000000
    MAX_VAL = MAX_N + 24
    
    # Precompute smooth parts using NumPy
    # smooth_part[x] = largest divisor of x composed only of primes < 24
    smooth_part = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in primes:
        # For each multiple of p, divide out all factors of p
        multiples = np.arange(p, MAX_VAL + 1, p)
        for m in multiples:
            while m % p == 0:
                smooth_part[m] *= p
                m //= p
    
    # Now search for n in [1000, 1000000] minimizing distinct smooth parts in n+1..n+24
    best_n = 1000
    best_count = float('inf')
    
    # Use a chunked approach with early termination
    # Process in chunks to check time periodically
    CHUNK = 10000
    for n_start in range(1000, MAX_N - 24 + 1, CHUNK):
        if time.time() - start_time > DEADLINE:
            break
        
        n_end = min(n_start + CHUNK, MAX_N - 24 + 1)
        
        for n in range(n_start, n_end):
            # Extract smooth parts for n+1 .. n+24
            window = smooth_part[n+1:n+25]
            # Count distinct values
            count = len(np.unique(window))
            
            if count < best_count:
                best_count = count
                best_n = n
                if best_count == 1:
                    return best_n
    
    return best_n
# EVOLVE-BLOCK-END
