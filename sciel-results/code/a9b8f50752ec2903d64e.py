"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    N = 10000000
    
    # Compute sigma(n) using SPF-based DP (more efficient than direct sieve)
    spf = np.zeros(N + 1, dtype=np.int32)
    spf[1] = 1
    for d in range(2, N + 1):
        if spf[d] == 0:
            spf[d::d] = d
        if (d & 0x3FFFF) == 0 and time.time() > deadline - 40:
            return [11, 6]
    
    if time.time() > deadline - 40:
        return [11, 6]
    
    # Compute sigma via DP on SPF
    sigma = np.ones(N + 1, dtype=np.int64)
    for n in range(2, N + 1):
        p = spf[n]
        m = n
        pe = 1
        s = 1
        while m % p == 0:
            m //= p
            pe *= p
            s += pe
        sigma[n] = sigma[m] * s
        if (n & 0x7FFFF) == 0 and time.time() > deadline - 25:
            return [11, 6]
    
    if time.time() > deadline - 25:
        return [11, 6]
    
    # Group indices by sigma value using numpy argsort
    indices = np.arange(1, N + 1, dtype=np.int32)
    sigma_vals = sigma[1:]
    
    order = np.argsort(sigma_vals, kind='stable')
    sorted_sigma = sigma_vals[order]
    sorted_indices = indices[order]
    
    # Find group boundaries
    change_points = np.where(np.diff(sorted_sigma) != 0)[0] + 1
    group_starts = np.concatenate([[0], change_points])
    group_ends = np.concatenate([change_points, [len(sorted_indices)]])
    
    if time.time() > deadline - 15:
        return [11, 6]
    
    # Find groups with size >= 2
    group_sizes = group_ends - group_starts
    valid_groups = np.where(group_sizes >= 2)[0]
    
    if len(valid_groups) == 0:
        return [11, 6]
    
    # Sort valid groups by size descending
    valid_groups = valid_groups[np.argsort(-group_sizes[valid_groups])]
    
    best_error = 1.0
    best_pair = (11, 6)
    
    for gi in valid_groups:
        if time.time() > deadline - 3:
            break
        
        s_start = group_starts[gi]
        s_end = group_ends[gi]
        nums_sorted = sorted_indices[s_start:s_end].astype(np.int64)
        n = len(nums_sorted)
        
        if n < 2:
            continue
        
        # Iterate b from large to small (larger b gives smaller error for same |2a-3b|)
        for i in range(n - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            b = int(nums_sorted[i])
            target_a = (3 * b) // 2
            pos = bisect_left(nums_sorted, target_a)
            
            candidates = []
            if pos < n:
                candidates.append(int(nums_sorted[pos]))
            if pos > 0:
                candidates.append(int(nums_sorted[pos - 1]))
            
            for a in candidates:
                if a == b:
                    continue
                error_num = abs(2 * a - 3 * b)
                if error_num == 0:
                    continue  # Invalid: exact 3/2 ratio
                error_den = 2 * b
                error = error_num / error_den
                if error < best_error:
                    best_error = error
                    best_pair = (a, b)
    
    a, b = best_pair
    a = int(a)
    b = int(b)
    if sigma[a] != sigma[b] or a == b:
        return [11, 6]
    return [a, b]

# EVOLVE-BLOCK-END
