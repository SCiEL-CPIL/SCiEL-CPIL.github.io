"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict

def construct():
    N_MAX = 1000000
    N_MIN = 1000
    WINDOW_SIZE = 24
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    M_start = N_MIN + 1
    M_end = N_MAX + WINDOW_SIZE
    size = M_end - M_start + 1
    
    S = np.ones(size, dtype=np.int64)
    
    # Precompute S(m) using vectorization
    for p in small_primes:
        power = p
        while power <= M_end:
            # Find indices j such that (M_start + j) is divisible by power
            # M_start + j = k * power => j = k * power - M_start
            # We need j in [0, size-1]
            # k_min = ceil(M_start / power)
            # k_max = floor(M_end / power)
            
            k_min = (M_start + power - 1) // power
            k_max = M_end // power
            
            if k_min <= k_max:
                # Indices j corresponding to k in [k_min, k_max]
                # j = k * power - M_start
                start_idx = k_min * power - M_start
                
                # Create array of indices
                idx = np.arange(start_idx, size, power)
                S[idx] *= p
                
            power *= p

    # Sliding window with frequency count
    freq = defaultdict(int)
    distinct_count = 0
    
    # Fill initial window [0, WINDOW_SIZE)
    for k in range(WINDOW_SIZE):
        v = S[k]
        if freq[v] == 0:
            distinct_count += 1
        freq[v] += 1
        
    min_distinct = distinct_count
    best_n = N_MIN
    
    # Slide window
    max_i = N_MAX - N_MIN
    for i in range(1, max_i + 1):
        # Remove S[i-1]
        v_out = S[i-1]
        freq[v_out] -= 1
        if freq[v_out] == 0:
            distinct_count -= 1
            
        # Add S[i+WINDOW_SIZE-1]
        v_in = S[i+WINDOW_SIZE-1]
        if freq[v_in] == 0:
            distinct_count += 1
        freq[v_in] += 1
        
        if distinct_count < min_distinct:
            min_distinct = distinct_count
            best_n = N_MIN + i
            
            if min_distinct == 1:
                break

    return best_n

# EVOLVE-BLOCK-END
