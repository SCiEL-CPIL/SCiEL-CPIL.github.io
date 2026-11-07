import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s

    N = 10_000_000
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Phase 2: i from N//2+1 to N, each contributes only to sigma[i]
    half = N // 2
    sigma[half + 1:] += np.arange(half + 1, N + 1, dtype=np.int64)
    
    # Phase 1: i from 1 to N//2, each has at least 2 multiples
    for i in range(1, half + 1):
        sigma[i::i] += i
        if i % 200000 == 0 and time.time() > deadline - 35.0:
            break
    
    if time.time() > deadline - 30.0:
        return [11, 6]
    
    # Now find pairs with equal sigma values
    sig_vals = sigma[1:]
    indices = np.arange(1, N + 1)
    
    # Sort by sigma value
    order = np.argsort(sig_vals, kind='mergesort')
    sorted_sig = sig_vals[order]
    sorted_idx = indices[order]
    
    # Find groups of equal sigma values
    changes = np.where(np.diff(sorted_sig) != 0)[0] + 1
    boundaries = np.concatenate(([0], changes, [len(sorted_sig)]))
    
    best_diff = float('inf')
    best_a = 11
    best_b = 6
    
    # Process groups with size >= 2, largest first for faster convergence
    group_sizes = np.diff(boundaries)
    valid_groups = np.where(group_sizes >= 2)[0]
    if len(valid_groups) > 0:
        valid_groups = valid_groups[np.argsort(group_sizes[valid_groups])[::-1]]
    
    for k in valid_groups:
        if time.time() > deadline - 2.0:
            break
        lo = boundaries[k]
        hi = boundaries[k + 1]
        
        group_idx = sorted_idx[lo:hi]
        n_group = len(group_idx)
        
        # Vectorized search: for each b, find a closest to 3b/2
        ideal = (3.0 * group_idx.astype(np.float64)) / 2.0
        pos = np.searchsorted(group_idx, ideal, side='left')
        
        for offset in [0, -1]:
            pa = pos + offset
            valid = (pa >= 0) & (pa < n_group)
            if not np.any(valid):
                continue
            pa_v = pa[valid]
            b_v = group_idx[valid]
            a_v = group_idx[pa_v]
            
            diff_num = np.abs(2 * a_v - 3 * b_v)
            mask = diff_num > 0
            if not np.any(mask):
                continue
            dn = diff_num[mask]
            bv = b_v[mask]
            av = a_v[mask]
            
            errs = dn.astype(np.float64) / (2.0 * bv.astype(np.float64))
            min_i = np.argmin(errs)
            if errs[min_i] < best_diff:
                best_diff = float(errs[min_i])
                best_a = int(av[min_i])
                best_b = int(bv[min_i])
    
    return [int(best_a), int(best_b)]
