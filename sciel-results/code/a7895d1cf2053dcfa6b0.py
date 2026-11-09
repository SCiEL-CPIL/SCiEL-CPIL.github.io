import time
import numpy as np

MAX_N = 10000000
TARGET_NUM = 3
TARGET_DEN = 2

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    # 1. Compute sigma(n)
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    for i in range(1, MAX_N + 1):
        sigma[i::i] += i
        
    if time.time() > deadline:
        return [11, 6]

    # 2. Sort and Group
    sorted_indices = np.argsort(sigma)
    sorted_sigmas = sigma[sorted_indices]
    
    if time.time() > deadline:
        return [11, 6]

    best_error = float('inf')
    best_pair = (11, 6)
    
    n = MAX_N
    i = 0
    
    while i < n:
        if time.time() > deadline:
            break
            
        current_sigma = sorted_sigmas[i]
        j = i + 1
        while j < n and sorted_sigmas[j] == current_sigma:
            j += 1
            
        if j - i >= 2:
            group_vals = sorted_indices[i:j]
            group_vals = np.sort(group_vals)
            
            p = 0
            m = len(group_vals)
            
            for b in group_vals:
                target = 1.5 * b
                while p < m and group_vals[p] < target:
                    p += 1
                    
                if p < m:
                    a = group_vals[p]
                    if a != b:
                        err_num = abs(2 * a - 3 * b)
                        if err_num > 0:
                            err_val = err_num / (2.0 * b)
                            if err_val < best_error:
                                best_error = err_val
                                best_pair = (int(a), int(b))
                
                if p > 0:
                    a = group_vals[p-1]
                    if a != b:
                        err_num = abs(2 * a - 3 * b)
                        if err_num > 0:
                            err_val = err_num / (2.0 * b)
                            if err_val < best_error:
                                best_error = err_val
                                best_pair = (int(a), int(b))
                                
        i = j

    if best_pair == (11, 6) and best_error == float('inf'):
        return [11, 6]
        
    a, b = best_pair
    return [a, b]
