"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    N = 1000000
    WINDOW = 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Compute the small-prime part for each number in [1, N+24]
    # small_part[m] = product of p^k for each small prime p where p^k || m
    size = N + WINDOW + 1  # need indices up to N + 24
    small_part = np.ones(size, dtype=np.int64)
    
    # Vectorized computation of small-prime parts
    for p in small_primes:
        # For each multiple of p, compute p^v_p(m)
        multiples = np.arange(p, size, p)
        # Compute the p-part for each multiple
        # For each m in multiples, we need p^k where k = v_p(m)
        # Vectorized: repeatedly divide by p
        vals = multiples.copy().astype(np.int64)
        p_power = np.ones(len(multiples), dtype=np.int64)
        
        # Count how many times p divides each value
        while np.any(vals % p == 0):
            mask = (vals % p == 0)
            p_power[mask] *= p
            vals[mask] //= p
        
        small_part[multiples] *= p_power
    
    # Now scan for the minimum distinct count in windows of 24
    # Use a sliding window with frequency counting
    from collections import defaultdict
    
    best_n = 1000
    best_distinct = WINDOW + 1  # worst case
    
    # Initialize window for n = 1000: indices 1001 to 1024
    freq = defaultdict(int)
    for i in range(1001, 1001 + WINDOW):
        val = small_part[i]
        freq[val] += 1
    
    distinct_count = len(freq)
    if distinct_count < best_distinct:
        best_distinct = distinct_count
        best_n = 1000
    
    # Early exit if we found the optimal
    if best_distinct == 1:
        return best_n
    
    # Slide the window
    for n in range(1001, N + 1):
        # Remove small_part[n] (the element leaving)
        old_val = small_part[n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
        
        # Add small_part[n + WINDOW] (the element entering)
        new_val = small_part[n + WINDOW]
        freq[new_val] += 1
        
        distinct_count = len(freq)
        if distinct_count < best_distinct:
            best_distinct = distinct_count
            best_n = n
            if best_distinct == 1:
                break
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return best_n

# EVOLVE-BLOCK-END
