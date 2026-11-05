import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin out of 90s
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_val = 1000025
    
    # Compute the 23-smooth part: for each x, the product of p^v_p(x) for p < 24
    # This is the "complete factor composed of primes STRICTLY BELOW 24"
    smooth_part = np.ones(max_val, dtype=np.int64)
    
    for p in primes:
        pk = p
        while pk < max_val:
            for j in range(pk, max_val, pk):
                smooth_part[j] *= p
            pk *= p
    
    if time.time() > deadline:
        return 1000
    
    min_n = 1000
    max_n = 1000000
    
    # First check for all-1 windows (distinct count = 1)
    is_smooth = (smooth_part == 1).astype(np.int8)
    cumsum = np.cumsum(is_smooth)
    
    s_min = min_n + 1
    s_max = max_n + 1
    s_arr = np.arange(s_min, s_max + 1)
    window_sum = cumsum[s_arr + 23] - cumsum[s_arr - 1]
    
    all_smooth_mask = (window_sum == 24)
    if np.any(all_smooth_mask):
        first_idx = np.argmax(all_smooth_mask)
        n_val = s_arr[first_idx] - 1
        return int(n_val)
    
    if time.time() > deadline:
        return 1000
    
    # No all-1 window. Find windows with minimal distinct count using vectorized batches
    best_count = 25
    best_n = min_n
    
    batch_size = 50000
    offsets = np.arange(1, 25)
    
    for batch_start in range(min_n, max_n + 1, batch_size):
        if time.time() > deadline:
            break
        
        batch_end = min(batch_start + batch_size, max_n + 1)
        n_vals = np.arange(batch_start, batch_end)
        
        indices = n_vals[:, None] + offsets[None, :]
        vals = smooth_part[indices]  # shape: (num_in_batch, 24)
        
        # Count distinct values in each row
        sorted_vals = np.sort(vals, axis=1)
        changes = np.diff(sorted_vals, axis=1) != 0
        distinct_counts = 1 + changes.sum(axis=1)
        
        min_idx = np.argmin(distinct_counts)
        min_count = int(distinct_counts[min_idx])
        
        if min_count < best_count:
            best_count = min_count
            best_n = int(n_vals[min_idx])
            if best_count == 2:
                break
    
    if time.time() > deadline:
        return int(best_n)
    
    # Random sampling for broader exploration
    remaining_time = deadline - time.time()
    if remaining_time > 1:
        num_random = min(100000, int(remaining_time * 20000))
        random_ns = np.random.randint(min_n, max_n + 1, size=num_random)
        
        for i in range(0, num_random, 1000):
            if time.time() > deadline:
                break
            
            chunk_size = min(1000, num_random - i)
            chunk_ns = random_ns[i:i+chunk_size]
            
            indices = chunk_ns[:, None] + offsets[None, :]
            vals = smooth_part[indices]
            
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
