"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Range: n in [1000, 1000000], so n+1 in [1001, 1000001], n+24 in [1024, 1000024]
    max_val = 1000024
    
    # Precompute 23-smooth part for all numbers up to max_val using vectorized ops
    # smooth_part[m] = the part of m composed only of primes < 24
    smooth_part = np.ones(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        power = p
        while power <= max_val:
            multiples_power = np.arange(power, max_val + 1, power)
            smooth_part[multiples_power] *= p
            power *= p
    
    # Now slide window: for each n in [1000, 1000000], look at n+1..n+24
    # Count distinct smooth parts
    best_n = 1000
    best_count = float('inf')
    
    n_start = 1000
    n_end = 1000000
    
    # Initial window: n=1000, numbers 1001..1024
    window_indices = np.arange(n_start + 1, n_start + 25)
    window_vals = smooth_part[window_indices]
    unique_vals, counts_arr = np.unique(window_vals, return_counts=True)
    counts = dict(zip(unique_vals, counts_arr))
    best_count = len(counts)
    best_n = n_start
    
    if best_count <= 1:
        return best_n
    
    # Sliding window optimization: use local variables for speed
    sp = smooth_part
    best_c = best_count
    best_n_local = best_n
    deadline_local = deadline
    t0 = time.time
    
    for n in range(n_start + 1, n_end + 1):
        out_val = n
        in_val = n + 24
        sp_out = sp[out_val]
        sp_in = sp[in_val]
        
        if sp_out != sp_in:
            c_out = counts[sp_out] - 1
            if c_out == 0:
                del counts[sp_out]
            else:
                counts[sp_out] = c_out
            counts[sp_in] = counts.get(sp_in, 0) + 1
        
        current_count = len(counts)
        if current_count < best_c:
            best_c = current_count
            best_n_local = n
            if best_c <= 1:
                break
        
        if t0() > deadline_local:
            break
    
    return best_n_local
# EVOLVE-BLOCK-END
