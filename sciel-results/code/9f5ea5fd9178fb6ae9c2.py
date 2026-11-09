import math
import time
import bisect
import numpy as np

MAX_N = 10_000_000
TIME_LIMIT = 50.0  # Safety margin

def construct():
    t_start = time.time()
    deadline = t_start + TIME_LIMIT

    # 1. Sieve sigma
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    for d in range(1, MAX_N + 1):
        sigma[d::d] += d

    # 2. Group indices by sigma value
    # Using a dictionary of lists is standard.
    # To speed up, we can use a list of lists if sigma values were small, but they are not.
    # Dict is fine.
    groups = {}
    sigma_list = sigma.tolist()
    
    # Optimization: Pre-allocate list sizes? No, hard to know.
    # Just append.
    for i in range(1, MAX_N + 1):
        s = sigma_list[i]
        if s in groups:
            groups[s].append(i)
        else:
            groups[s] = [i]
            
    # Sort each group
    for key in groups:
        if len(groups[key]) > 1:
            groups[key].sort()
            
    # 3. Search
    best_err = float('inf')
    best_pair = (1, 1) # Fallback
    
    bisect_left = bisect.bisect_left
    
    # Iterate b from MAX_N down to 1
    # We can stop if 1/(2*b) >= best_err
    # Note: best_err is updated as we go.
    
    for b in range(MAX_N, 0, -1):
        if time.time() > deadline:
            break
            
        s = sigma_list[b]
        lst = groups.get(s)
        if not lst or len(lst) < 2:
            continue
            
        n = len(lst)
        target = 1.5 * b
        
        # Binary search
        pos = bisect_left(lst, target)
        
        # Check candidates
        # pos and pos-1
        if pos < n:
            a = lst[pos]
            if a != b:
                num = abs(2*a - 3*b)
                if num > 0:
                    err = num / (2.0 * b)
                    if err < best_err:
                        best_err = err
                        best_pair = (a, b)
                        
        if pos > 0:
            a = lst[pos-1]
            if a != b:
                num = abs(2*a - 3*b)
                if num > 0:
                    err = num / (2.0 * b)
                    if err < best_err:
                        best_err = err
                        best_pair = (a, b)
                        
        # Check break condition
        # If min possible error for any b' < b is >= best_err, stop.
        # Min error for b' is approx 1/(2*b').
        # For b' = b-1, min err is 1/(2*(b-1)).
        # If 1/(2*b) >= best_err, then for all b' <= b, 1/(2*b') >= 1/(2*b) >= best_err.
        # So we can break.
        if best_err > 0 and (1.0 / (2.0 * b)) >= best_err:
            break

    a, b = best_pair
    return [int(a), int(b)]
