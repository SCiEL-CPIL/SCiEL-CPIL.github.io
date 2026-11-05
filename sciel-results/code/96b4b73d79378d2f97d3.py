import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin out of 90s
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_val = 1000025
    
    # Precompute small part for each number 1..max_val-1
    # small_part[j] = j with all factors of primes in `primes` removed
    small_part = np.arange(max_val, dtype=np.int64)
    
    for p in primes:
        pk = p
        while pk < max_val:
            for j in range(pk, max_val, pk):
                small_part[j] //= p
            pk *= p
    
    if time.time() > deadline:
        return 1000
    
    min_n = 1000
    max_n = 1000000
    n_count = max_n - min_n + 1  # 999001
    
    # For each n in [min_n, max_n], window is small_part[n+1 : n+25]
    # We want to minimize the number of distinct values in each window
    
    # Strategy: 
    # 1. First, check for all-1 windows (distinct count = 1)
    # 2. Then, use vectorized batched evaluation for remaining windows
    # 3. Add random sampling for broader exploration
    
    # Check for all-1 windows: small_part[n+1:n+25] all == 1
    # This means n+1, ..., n+24 are all 23-smooth
    is_smooth = (small_part == 1).astype(np.int8)
    
    # Sliding window sum of is_smooth
    # Window starting at index s (where s = n+1) has sum 24 if all smooth
    # s ranges from min_n+1 to max_n+1
    # We need sum of is_smooth[s:s+24] == 24
    
    cumsum = np.cumsum(is_smooth)
    # For s in [min_n+1, max_n+1]:
    # sum = cumsum[s+23] - cumsum[s-1]
    
    s_min = min_n + 1
    s_max = max_n + 1
    s_arr = np.arange(s_min, s_max + 1)
    
    # Handle boundary: s-1 could be 0
    # cumsum[s+23] - cumsum[s-1]
    # s+23 ranges from min_n+24 to max_n+24
    # s-1 ranges from min_n to max_n
    
    window_sum = cumsum[s_arr + 23] - cumsum[s_arr - 1]
    
    all_smooth_mask = (window_sum == 24)
    if np.any(all_smooth_mask):
        # Found a window with all lp == 1, distinct count = 1
        first_idx = np.argmax(all_smooth_mask)
        n_val = s_arr[first_idx] - 1
        return int(n_val)
    
    if time.time() > deadline:
        return 1000
    
    # No all-1 window. Now find windows with minimal distinct count.
    # Use batched vectorized evaluation
    
    best_count = 25
    best_n = min_n
    
    # Batch evaluation: process windows in batches
    batch_size = 50000
    offsets = np.arange(1, 25)  # 1 to 24
    
    for batch_start in range(min_n, max_n + 1, batch_size):
        if time.time() > deadline:
            break
        
        batch_end = min(batch_start + batch_size, max_n + 1)
        n_vals = np.arange(batch_start, batch_end)
        num_in_batch = len(n_vals)
        
        # indices[i, j] = n_vals[i] + offsets[j]
        indices = n_vals[:, None] + offsets[None, :]
        vals = small_part[indices]  # shape: (num_in_batch, 24)
        
        # Count distinct values in each row
        # Sort each row, then count changes
        sorted_vals = np.sort(vals, axis=1)
        changes = np.diff(sorted_vals, axis=1) != 0
        distinct_counts = 1 + changes.sum(axis=1)
        
        min_idx = np.argmin(distinct_counts)
        min_count = int(distinct_counts[min_idx])
        
        if min_count < best_count:
            best_count = min_count
            best_n = int(n_vals[min_idx])
            if best_count == 2:
                # Can't do better than 2 without all-1 (which we already checked)
                break
    
    if time.time() > deadline:
        return int(best_n)
    
    # Additional random sampling to explore the space more broadly
    # Sample random windows and check their distinct counts
    num_random = 100000
    remaining_time = deadline - time.time()
    time_per_sample = remaining_time / num_random
    
    random_ns = np.random.randint(min_n, max_n + 1, size=num_random)
    
    for i in range(0, num_random, 1000):
        if time.time() > deadline:
            break
        
        chunk_size = min(1000, num_random - i)
        chunk_ns = random_ns[i:i+chunk_size]
        
        indices = chunk_ns[:, None] + offsets[None, :]
        vals = small_part[indices]
        
        sorted_vals = np.sort(vals, axis=1)
        changes = np.diff(sorted_vals, axis=1) != 0
        distinct_counts = 1 + changes.sum(axis=1)
        
        min_idx = np.argmin(distinct_counts)
        min_count = int(distinct_counts[min_idx])
        
        if min_count < best_count:
            best_count = min_count
            best_n = int(chunk_ns[min_idx])
            if best_count == 2:
                break
    
    return int(best_n)
