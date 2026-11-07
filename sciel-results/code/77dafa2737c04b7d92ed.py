import numpy as np
import time
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill
    
    # Start with a manageable N, increase if time permits
    # The sieve is O(N log N), search is O(N log N) in worst case
    N = 5000000  # Start with 5M to be safe
    
    # Compute sigma(n) for all n up to N using sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve for sigma
    for i in range(1, N + 1):
        if time.time() > deadline - 20.0:
            N = i - 1
            break
        sigma[i::i] += i
    
    if N < 100:
        return [11, 6]
    
    if time.time() > deadline - 15.0:
        return [11, 6]
    
    # Sort by sigma value
    indices = np.arange(1, N + 1, dtype=np.int32)
    order = np.argsort(sigma[1:], kind='stable')
    sorted_sigma = sigma[1:][order]
    sorted_n = indices[order]
    
    if time.time() > deadline - 10.0:
        return [11, 6]
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    i = 0
    total_n = N
    while i < total_n:
        if time.time() > deadline - 3.0:
            break
        
        j = i + 1
        while j < total_n and sorted_sigma[j] == sorted_sigma[i]:
            j += 1
        
        group_size = j - i
        if group_size >= 2:
            group = sorted_n[i:j]
            # group is already sorted by n
            
            # For efficiency, only check a limited number of b values
            # Focus on b values where 3b/2 is likely to be in the group
            # Sample b values if group is large
            if group_size > 1000:
                # Sample evenly spaced b values
                step = max(1, group_size // 200)
                b_indices = range(0, group_size, step)
            else:
                b_indices = range(group_size)
            
            for b_idx in b_indices:
                b_val = int(group[b_idx])
                target_a = 3 * b_val / 2.0
                
                # Binary search in group for target_a
                pos = bisect_left(group, target_a)
                
                candidates = []
                if pos < group_size:
                    candidates.append(int(group[pos]))
                if pos > 0:
                    candidates.append(int(group[pos - 1]))
                # Also check pos+1 and pos-2 for safety
                if pos + 1 < group_size:
                    candidates.append(int(group[pos + 1]))
                if pos - 2 >= 0:
                    candidates.append(int(group[pos - 2]))
                
                for a_val in candidates:
                    if a_val == b_val:
                        continue
                    error_num = abs(2 * a_val - 3 * b_val)
                    if error_num == 0:
                        continue
                    error = error_num / (2 * b_val)
                    if error < best_error:
                        best_error = error
                        best_a = a_val
                        best_b = b_val
        
        i = j
    
    if best_error == float('inf'):
        return [11, 6]
    
    return [best_a, best_b]
