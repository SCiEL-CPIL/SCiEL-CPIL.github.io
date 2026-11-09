import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    n_min = 1000
    n_max = 1000000
    
    val_min = n_min + 1
    val_max = n_max + 24
    
    size = n_max - n_min + 1
    
    # Precompute G using efficient vectorized operations
    indices = np.arange(val_min, val_max + 1, dtype=np.int64)
    G = np.ones(len(indices), dtype=np.int64)
    
    for p in primes:
        p_power = p
        while p_power <= val_max:
            # Use np.where for potentially faster masking
            mask = (indices % p_power == 0)
            G[mask] *= p
            p_power *= p
            if time.time() > deadline:
                return 1000
    
    # Convert to list for faster Python loop access
    G_list = G.tolist()
    
    freq = defaultdict(int)
    
    # Initialize first window
    active_count = 0
    for i in range(24):
        v = G_list[i]
        if freq[v] == 0:
            active_count += 1
        freq[v] += 1
        
    min_distinct = active_count
    best_n = n_min
    
    # Slide window using list for faster access
    for i in range(1, size):
        # Remove G[i-1]
        val_out = G_list[i-1]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            active_count -= 1
            
        # Add G[i+23]
        val_in = G_list[i+23]
        if freq[val_in] == 0:
            active_count += 1
        freq[val_in] += 1
        
        if active_count < min_distinct:
            min_distinct = active_count
            best_n = n_min + i
            
        if i % 100000 == 0 and time.time() > deadline:
            break
            
    return int(best_n)
