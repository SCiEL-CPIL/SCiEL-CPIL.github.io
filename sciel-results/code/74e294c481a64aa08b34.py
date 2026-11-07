import time
import numpy as np
import sys

def construct():
    start_time = time.time()
    N = 10_000_000
    
    # Step 1: Compute sigma(n) for n=1..N using a multiplicative sieve
    # Use smallest prime factor (SPF) to compute sigma via DP.
    
    # Compute SPF using vectorized sieve
    spf = np.zeros(N + 1, dtype=np.int64)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i::i] = i
    
    # Compute sigma via DP using arrays
    # h[n] = highest power of spf[n] dividing n
    # g[n] = 1 + spf[n] + ... + spf[n]^k where spf[n]^k || n  
    # r[n] = n // spf[n]^k
    # sigma[n] = sigma[r[n]] * g[n]
    
    h = np.zeros(N + 1, dtype=np.int64)
    g = np.zeros(N + 1, dtype=np.int64)
    r = np.zeros(N + 1, dtype=np.int64)
    sigma_arr = np.zeros(N + 1, dtype=np.int64)
    
    h[1] = 1
    g[1] = 1
    r[1] = 1
    sigma_arr[1] = 1
    
    # Use Python lists for speed in the DP loop (avoid NumPy scalar overhead)
    spf_list = spf.tolist()
    h_list = h.tolist()
    g_list = g.tolist()
    r_list = r.tolist()
    sigma_list = sigma_arr.tolist()
    
    for n in range(2, N + 1):
        if time.time() - start_time > 25:
            break
        p = spf_list[n]
        m = n // p
        if m % p != 0:
            h_list[n] = p
            g_list[n] = 1 + p
            r_list[n] = m
        else:
            h_list[n] = h_list[m] * p
            g_list[n] = g_list[m] + h_list[n]
            r_list[n] = r_list[m]
        
        sigma_list[n] = sigma_list[r_list[n]] * g_list[n]
    
    sigma = np.array(sigma_list, dtype=np.int64)
    
    # Check deadline
    if time.time() - start_time > 35:
        return [11, 6]
    
    # Step 2: Group indices by sigma value
    indices = np.arange(1, N + 1)
    
    order = np.argsort(sigma[1:N+1])
    sorted_sigmas = sigma[1:N+1][order]
    sorted_indices = indices[order]
    
    # Find group boundaries
    changes = np.diff(sorted_sigmas) != 0
    boundaries = np.concatenate(([0], np.where(changes)[0] + 1, [N]))
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # Sort groups by size descending to find best pairs in large groups first
    group_sizes = np.diff(boundaries)
    large_groups = np.where(group_sizes > 1)[0]
    if len(large_groups) > 0:
        large_groups = large_groups[np.argsort(-group_sizes[large_groups])]
    
    # Iterate over groups, starting with largest
    for i in large_groups:
        start = boundaries[i]
        end = boundaries[i+1]
        size = end - start
        
        if size <= 1:
            continue
        
        # Check deadline
        if time.time() - start_time > 55:
            break
            
        # Extract group indices (already sorted by sigma, but we need them sorted by value)
        group_indices = np.sort(sorted_indices[start:end])
        
        # We want to find a, b in group_indices such that |a/b - 1.5| is minimized.
        # For each b, a should be close to 1.5 * b.
        
        # Vectorized approach for the group:
        targets = 1.5 * group_indices.astype(np.float64)
        pos = np.searchsorted(group_indices, targets, side='left')
        
        # Vectorized candidate extraction
        mask1 = pos < size
        mask2 = pos > 0
        
        # Combine both candidates and check
        # For cand1: a = group_indices[pos], b = group_indices
        # For cand2: a = group_indices[pos-1], b = group_indices
        
        # Check cand1
        if np.any(mask1):
            a_vals = group_indices[pos[mask1]]
            b_vals = group_indices[mask1]
            valid = a_vals != b_vals
            if np.any(valid):
                a_v = a_vals[valid]
                b_v = b_vals[valid]
                errors = np.abs(a_v / b_v - 1.5)
                min_idx = np.argmin(errors)
                min_err = errors[min_idx]
                if min_err < best_error:
                    best_error = min_err
                    best_a = int(a_v[min_idx])
                    best_b = int(b_v[min_idx])
        
        # Check cand2
        if np.any(mask2):
            a_vals = group_indices[pos[mask2] - 1]
            b_vals = group_indices[mask2]
            valid = a_vals != b_vals
            if np.any(valid):
                a_v = a_vals[valid]
                b_v = b_vals[valid]
                errors = np.abs(a_v / b_v - 1.5)
                min_idx = np.argmin(errors)
                min_err = errors[min_idx]
                if min_err < best_error:
                    best_error = min_err
                    best_a = int(a_v[min_idx])
                    best_b = int(b_v[min_idx])
    
    return [int(best_a), int(best_b)]
