import numpy as np
import time
from fractions import Fraction

def construct():
    t_start = time.time()
    deadline = t_start + 82.0
    
    N = 10_000_000
    
    # 1. Compute sigma(n) for n in [1, N] using a sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    if time.time() > deadline:
        return [1, 1]
    
    # 2. Sort indices by sigma value, then by index
    indices = np.arange(1, N + 1)
    keys = sigma[indices]
    order = np.lexsort((indices, keys))
    sorted_n = indices[order]
    sorted_s = keys[order]
    
    # 3. Build mapping from sigma value to range in sorted_n
    sigma_to_range = {}
    current_s = sorted_s[0]
    start_idx = 0
    for i in range(1, len(sorted_s)):
        if sorted_s[i] != current_s:
            sigma_to_range[current_s] = (start_idx, i)
            current_s = sorted_s[i]
            start_idx = i
    sigma_to_range[current_s] = (start_idx, len(sorted_s))
    
    if time.time() > deadline:
        return [1, 1]
    
    # 4. Iterate b to find best pair
    best_err = float('inf')
    best_a = 1
    best_b = 1
    
    for b in range(1, N + 1):
        if time.time() > deadline - 1.0:
            break
            
        s = sigma[b]
        if s not in sigma_to_range:
            continue
            
        start, end = sigma_to_range[s]
        sub_n = sorted_n[start:end]
        
        if len(sub_n) == 1:
            continue
            
        target = 1.5 * b
        
        pos = np.searchsorted(sub_n, target)
        
        candidates = []
        if pos > 0:
            candidates.append(sub_n[pos-1])
        if pos < len(sub_n):
            candidates.append(sub_n[pos])
        # Also check neighbors for safety
        if pos + 1 < len(sub_n):
            candidates.append(sub_n[pos+1])
        if pos - 2 >= 0:
            candidates.append(sub_n[pos-2])
        
        for a in candidates:
            a_int = int(a)
            if a_int == b:
                continue
            # Check for exact 3/2 ratio
            if 2 * a_int == 3 * b:
                continue
            # Calculate error
            err = abs(a_int / b - 1.5)
            if err < best_err:
                best_err = err
                best_a = a_int
                best_b = b
                
    return [int(best_a), int(best_b)]
