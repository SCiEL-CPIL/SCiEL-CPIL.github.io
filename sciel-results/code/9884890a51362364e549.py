import numpy as np
import math
import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 30
    L = 1
    for i in range(1, n + 1):
        L = L * i // math.gcd(L, i)
        
    weights = np.array([L // i for i in range(1, n + 1)], dtype=np.int64)
    
    n1 = 15
    n2 = 15
    
    weights1 = weights[:n1]
    weights2 = weights[n1:]
    
    def gen_sums_numpy(weights):
        m = len(weights)
        total = 3 ** m
        # Generate all indices
        indices = np.arange(total, dtype=np.int64)
        
        # Decode base 3 digits
        # digits shape: (total, m)
        digits = np.zeros((total, m), dtype=np.int8)
        base = 3
        for i in range(m):
            digits[:, i] = (indices // (base ** i)) % base
            # indices //= base ** i  # Not needed if we compute directly
        
        # Convert -1, 0, 1
        coeffs = digits - 1
        
        # Compute sums: coeffs @ weights
        sums = coeffs @ weights
        return sums, indices

    sums1, indices1 = gen_sums_numpy(weights1)
    sums2, indices2 = gen_sums_numpy(weights2)
    
    # Sort
    order1 = np.argsort(sums1)
    order2 = np.argsort(sums2)
    
    sorted_sums1 = sums1[order1]
    sorted_indices1 = indices1[order1]
    
    sorted_sums2 = sums2[order2]
    sorted_indices2 = indices2[order2]
    
    best_val = None
    best_idx1 = None
    best_idx2 = None
    
    # Search
    # We can vectorize the search?
    # For each a in sorted_sums1, we want b in sorted_sums2 closest to -a.
    # np.searchsorted can do this for all a at once.
    
    targets = -sorted_sums1
    insert_pts = np.searchsorted(sorted_sums2, targets)
    
    # Clamp insert_pts
    insert_pts = np.clip(insert_pts, 0, len(sorted_sums2) - 1)
    
    # Candidate 1: sorted_sums2[insert_pts]
    b1 = sorted_sums2[insert_pts]
    s1 = sorted_sums1 + b1
    idx2_1 = insert_pts
    
    # Candidate 2: sorted_sums2[insert_pts - 1]
    # Handle index 0
    insert_pts_minus = np.clip(insert_pts - 1, 0, len(sorted_sums2) - 1)
    b2 = sorted_sums2[insert_pts_minus]
    s2 = sorted_sums1 + b2
    idx2_2 = insert_pts_minus
    
    # Combine candidates
    # We need to find min |s| != 0
    # Create arrays of candidates
    # s1 and s2 are arrays of length len(sorted_sums1)
    
    # Mask out zeros
    mask1 = s1 != 0
    mask2 = s2 != 0
    
    # We want the global minimum absolute value across both sets
    # Flatten and search?
    # Or just take min of min(abs(s1[mask1])) and min(abs(s2[mask2]))
    
    min_val_1 = np.inf
    min_val_2 = np.inf
    min_idx_1 = -1
    min_idx_2 = -1
    
    if np.any(mask1):
        abs_s1 = np.abs(s1)
        abs_s1[~mask1] = np.inf
        min_idx_1 = np.argmin(abs_s1)
        min_val_1 = abs_s1[min_idx_1]
        
    if np.any(mask2):
        abs_s2 = np.abs(s2)
        abs_s2[~mask2] = np.inf
        min_idx_2 = np.argmin(abs_s2)
        min_val_2 = abs_s2[min_idx_2]
        
    if min_val_1 < min_val_2:
        best_val = min_val_1
        best_idx1 = sorted_indices1[min_idx_1]
        best_idx2 = sorted_indices2[idx2_1[min_idx_1]]
    else:
        best_val = min_val_2
        best_idx1 = sorted_indices1[min_idx_2]
        best_idx2 = sorted_indices2[idx2_2[min_idx_2]]
        
    if best_val is None or best_val == np.inf:
        return [0] * 28 + [1, -1]
        
    def decode_idx(idx, m):
        coeffs = np.zeros(m, dtype=np.int8)
        k = int(idx)
        for i in range(m):
            coeffs[i] = (k % 3) - 1
            k //= 3
        return coeffs.tolist()
    
    c1 = decode_idx(best_idx1, n1)
    c2 = decode_idx(best_idx2, n2)
    
    return c1 + c2
