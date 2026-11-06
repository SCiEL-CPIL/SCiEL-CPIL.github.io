def construct():
    import time
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N_MAX = 1000000
    # We need lpf up to N_MAX + 1 because we check n+1
    limit = N_MAX + 1
    
    # Initialize largest prime factor array
    # Using int32 is sufficient for values up to 1e6
    lpf = np.zeros(limit + 1, dtype=np.int32)
    
    # Sieve for largest prime factor
    # i is the current candidate for prime
    for i in range(2, limit + 1):
        if lpf[i] == 0:
            # i is prime
            # Update all multiples of i
            # Since we iterate i in increasing order, the last prime that divides j
            # will be the largest prime factor of j.
            for j in range(i, limit + 1, i):
                lpf[j] = i
        
        # Check time periodically
        if i % 10000 == 0 and time.time() > deadline:
            break

    # Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    # n is in [10000, 1000000]
    # n+1 is in [10001, 1000001]
    
    n_start = 10000
    n_end = N_MAX
    
    # Create array of n values
    # We can just compute the max array and find min
    # n_indices = np.arange(n_start, n_end + 1)
    # vals = np.maximum(lpf[n_indices], lpf[n_indices + 1])
    # min_idx = np.argmin(vals)
    # best_n = n_indices[min_idx]
    
    # To save memory and time, we can do it in chunks or just vectorized.
    # Vectorized is fine for 1e6 elements.
    n_indices = np.arange(n_start, n_end + 1, dtype=np.int32)
    vals = np.maximum(lpf[n_indices], lpf[n_indices + 1])
    min_idx = np.argmin(vals)
    best_n = int(n_indices[min_idx])
    
    return best_n
