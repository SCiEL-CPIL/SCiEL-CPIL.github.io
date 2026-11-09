import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    n_min = 1000
    n_max = 1000000
    
    val_min = n_min + 1
    val_max = n_max + 24
    
    size = n_max - n_min + 1  # number of n values
    total_vals = val_max - val_min + 1  # number of values to compute G for
    
    # Precompute G using vectorized operations
    indices = np.arange(val_min, val_max + 1, dtype=np.int64)
    G = np.ones(total_vals, dtype=np.int64)
    
    for p in primes:
        p_power = p
        while p_power <= val_max:
            mask = (indices % p_power == 0)
            G[mask] *= p
            p_power *= p
            if time.time() > deadline:
                return 1000
    
    # Map G values to small integer indices for faster dictionary operations
    unique_G, inverse = np.unique(G, return_inverse=True)
    num_unique = len(unique_G)
    
    # Build frequency array for sliding window
    freq = np.zeros(num_unique, dtype=np.int32)
    
    # Initialize first window: positions 0..23 in G correspond to n_min+1..n_min+24
    window_start = 0
    window_end = 23  # inclusive, so 24 elements
    for i in range(window_end - window_start + 1):
        freq[inverse[i]] += 1
    
    active_count = np.count_nonzero(freq)
    
    min_distinct = active_count
    best_n = n_min
    
    # Slide window using vectorized count of non-zero frequencies
    for i in range(1, size):
        # Remove element at position i-1, add element at position i+23
        freq[inverse[i - 1]] -= 1
        freq[inverse[i + 23]] += 1
        
        # Count distinct: number of non-zero entries in freq
        new_active = np.count_nonzero(freq)
        
        if new_active < min_distinct:
            min_distinct = new_active
            best_n = n_min + i
            if min_distinct <= 2:  # Can't do much better than 2
                break
        
        if i % 50000 == 0 and time.time() > deadline:
            break
    
    return int(best_n)
