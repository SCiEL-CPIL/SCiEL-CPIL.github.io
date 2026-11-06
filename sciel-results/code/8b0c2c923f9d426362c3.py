"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Range: n in [1000, 1000000], so n+1 in [1001, 1000001], n+24 in [1024, 1000024]
    # We need smooth parts for numbers from 1001 to 1000024
    max_val = 1000024
    
    # Precompute 23-smooth part for all numbers up to max_val
    # smooth_part[m] = the part of m composed only of primes < 24
    smooth_part = np.zeros(max_val + 1, dtype=np.int64)
    smooth_part[0] = 0
    smooth_part[1] = 1
    
    for m in range(2, max_val + 1):
        x = m
        sp = 1
        for p in small_primes:
            while x % p == 0:
                x //= p
                sp *= p
        smooth_part[m] = sp
    
    # Now slide window: for each n in [1000, 1000000], look at n+1..n+24
    # Count distinct smooth parts
    best_n = 1000
    best_count = float('inf')
    
    # Use a sliding window approach with a counter
    # For efficiency, process in chunks
    n_start = 1000
    n_end = 1000000
    
    # Initial window: n=1000, numbers 1001..1024
    from collections import Counter
    
    window = list(range(n_start + 1, n_start + 25))
    counts = Counter(smooth_part[m] for m in window)
    best_count = len(counts)
    best_n = n_start
    
    for n in range(n_start + 1, n_end + 1):
        # Remove n (the leftmost), add n+24 (the rightmost)
        out_val = n
        in_val = n + 24
        sp_out = smooth_part[out_val]
        sp_in = smooth_part[in_val]
        
        counts[sp_out] -= 1
        if counts[sp_out] == 0:
            del counts[sp_out]
        counts[sp_in] += 1
        
        current_count = len(counts)
        if current_count < best_count:
            best_count = current_count
            best_n = n
            if best_count <= 2:
                break  # Can't do much better than 1 or 2
        
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
