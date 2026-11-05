import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin out of 90s
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_val = 1000025
    
    # Precompute small part for each number 1..max_val
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
    
    # For each n in [min_n, max_n], the window is small_part[n+1 : n+25]
    # s = n+1, so s ranges from min_n+1 to max_n+1
    # We need small_part[s : s+24] for s in [min_n+1, max_n+1]
    
    # Cheap proxy: count of 1s in each window (number of 23-smooth numbers)
    is_one = (small_part == 1).astype(np.int8)
    
    # Compute sliding window sum of is_one for window size 24
    # For s in [min_n+1, max_n+1], window sum = sum(is_one[s:s+24])
    # Using cumsum:
    cumsum = np.zeros(len(is_one) + 1, dtype=np.int32)
    np.cumsum(is_one, out=cumsum[1:])
    
    # s ranges from min_n+1 to max_n+1
    # sum(is_one[s:s+24]) = cumsum[s+24] - cumsum[s]
    s_start = min_n + 1
    s_end = max_n + 1  # inclusive
    # s+24 ranges from s_start+24 to s_end+24
    # cumsum indices: s to s+24, so we need cumsum[s+24] - cumsum[s]
    
    one_counts = cumsum[s_start + 24 : s_end + 24 + 1] - cumsum[s_start : s_end + 1]
    # one_counts[i] corresponds to n = min_n + i
    
    if time.time() > deadline:
        # Fallback: just return the n with most 1s
        best_idx = np.argmax(one_counts)
        return int(min_n + best_idx)
    
    # Select top K candidates based on one_counts (higher is better)
    # Also consider: windows where the max small_part value is small
    # Let's also compute max small_part in each window as a secondary criterion
    
    # Compute sliding window max of small_part
    # This is harder with cumsum, so let's use a different approach
    # For now, just use one_counts as the primary filter
    
    # Get top K candidates
    K = 50000
    if n_count > K:
        # Get indices of top K one_counts
        # Use argpartition for O(N) selection
        top_indices = np.argpartition(one_counts, -K)[-K:]
    else:
        top_indices = np.arange(n_count)
    
    # Sort these candidates by one_counts descending for better ordering
    top_indices = top_indices[np.argsort(one_counts[top_indices])[::-1]]
    
    best_count = 25
    best_n = min_n + top_indices[0]
    
    # Evaluate exact distinct count for top candidates
    for idx in top_indices:
        if time.time() > deadline:
            break
        n = min_n + idx
        vals = small_part[n+1 : n+25]
        distinct_count = len(np.unique(vals))
        
        if distinct_count < best_count:
            best_count = int(distinct_count)
            best_n = int(n)
            if best_count == 1:
                return best_n
    
    # If we haven't found a good answer, do a broader scan with the batched approach
    if best_count > 3 and time.time() < deadline - 5:
        # Fallback: batched scan over remaining candidates
        # This is the original approach but only for windows not yet checked
        # For simplicity, just do a full batched scan
        batch_size = 100000
        for batch_start in range(min_n, max_n + 1, batch_size):
            if time.time() > deadline - 2:
                break
            batch_end = min(batch_start + batch_size, max_n + 1)
            n_vals = np.arange(batch_start, batch_end)
            offsets = np.arange(1, 25)
            indices = n_vals[:, None] + offsets[None, :]
            vals = small_part[indices]
            
            sorted_vals = np.sort(vals, axis=1)
            changes = np.diff(sorted_vals, axis=1) != 0
            distinct_counts = 1 + changes.sum(axis=1)
            
            min_idx = np.argmin(distinct_counts)
            if distinct_counts[min_idx] < best_count:
                best_count = int(distinct_counts[min_idx])
                best_n = int(n_vals[min_idx])
                if best_count == 1:
                    return best_n
    
    return int(best_n)
