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
    
    best_count = 25
    best_n = 1000
    
    min_n = 1000
    max_n = 1000000
    batch_size = 100000
    
    for batch_start in range(min_n, max_n + 1, batch_size):
        if time.time() > deadline:
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
    
    return best_n
