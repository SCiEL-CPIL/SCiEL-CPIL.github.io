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
    
    # Precompute smooth part using iterative division (faster than multiplication approach)
    # For each number, repeatedly divide by each small prime to get the rough part
    # smooth_part = original / rough_part
    arr = np.arange(1, max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        # Repeatedly divide by p
        mask = (arr % p == 0)
        while np.any(mask):
            arr[mask] //= p
            mask = (arr % p == 0)
    
    # arr now holds the rough part (m with all small prime factors removed)
    # smooth_part[m] = m_original / arr[m]
    original = np.arange(1, max_val + 1, dtype=np.int64)
    smooth_part = original // arr
    
    # Map each distinct smooth part to a compact integer ID
    unique_sp, inverse = np.unique(smooth_part, return_inverse=True)
    sp_ids = inverse  # sp_ids[i] is the ID for smooth_part[i]
    num_distinct = len(unique_sp)
    
    # Now slide window: for each n in [1000, 1000000], look at n+1..n+24
    # Count distinct smooth part IDs in the window
    best_n = 1000
    best_count = float('inf')
    
    n_start = 1000
    n_end = 1000000
    
    # Use a frequency array indexed by sp_id for O(1) updates
    freq = np.zeros(num_distinct, dtype=np.int32)
    
    # Initial window: n=1000, numbers 1001..1024
    # sp_ids[m-1] is the ID for number m (since sp_ids[0] corresponds to number 1)
    for m in range(n_start + 1, n_start + 25):
        freq[sp_ids[m - 1]] += 1
    
    current_count = np.count_nonzero(freq)
    best_count = current_count
    best_n = n_start
    
    for n in range(n_start + 1, n_end + 1):
        # Remove n (the leftmost), add n+24 (the rightmost)
        out_id = sp_ids[n - 1]
        in_id = sp_ids[n + 24 - 1]
        
        freq[out_id] -= 1
        if freq[out_id] == 0:
            current_count -= 1
        
        freq[in_id] += 1
        if freq[in_id] == 1:
            current_count += 1
        
        if current_count < best_count:
            best_count = current_count
            best_n = n
            if best_count <= 2:
                break
        
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
