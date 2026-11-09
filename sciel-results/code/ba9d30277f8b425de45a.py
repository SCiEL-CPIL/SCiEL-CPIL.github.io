import numpy as np
import time
from bisect import bisect_left

def construct():
    N = 10000000
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Compute sigma(n) for all n <= N
    sigma = np.ones(N + 1, dtype=np.int64)
    for i in range(2, N + 1):
        sigma[i::i] += i
    
    if time.time() > deadline:
        return [11, 6]
    
    # Sort by sigma value
    idx = np.arange(1, N + 1)
    sigma_vals = sigma[1:]
    order = np.argsort(sigma_vals, kind='stable')
    sorted_idx = idx[order]
    sorted_sigma = sigma_vals[order]
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    n_total = len(sorted_idx)
    i = 0
    while i < n_total:
        j = i + 1
        while j < n_total and sorted_sigma[j] == sorted_sigma[i]:
            j += 1
        
        group_size = j - i
        if group_size >= 2:
            group = np.sort(sorted_idx[i:j])
            group_list = group.tolist()
            
            for k in range(group_size):
                b = group_list[k]
                target = 1.5 * b
                pos = bisect_left(group_list, target)
                
                for p in (pos, pos - 1):
                    if 0 <= p < group_size:
                        a = group_list[p]
                        if a != b:
                            D = abs(2 * a - 3 * b)
                            if D > 0:
                                error = D / (2.0 * b)
                                if error < best_error:
                                    best_error = error
                                    best_a = a
                                    best_b = b
        
        i = j
        if time.time() > deadline:
            break
    
    return [int(best_a), int(best_b)]
