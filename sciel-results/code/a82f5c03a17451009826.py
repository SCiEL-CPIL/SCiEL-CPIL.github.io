import time
import numpy as np
import sys

def construct():
    start_time = time.time()
    N = 10_000_000
    
    # Step 1: Compute sigma(n) for n=1..N
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # Check deadline
    if time.time() - start_time > 40:
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
    
    # Iterate over groups
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        size = end - start
        
        if size <= 1:
            continue
            
        # Extract group indices
        group_indices = sorted_indices[start:end]
        
        # Sort the group indices to allow binary search
        group_indices = np.sort(group_indices)
        
        # We want to find a, b in group_indices such that |a/b - 1.5| is minimized.
        # This is equivalent to minimizing |2a - 3b|.
        # For each b, a should be close to 1.5 * b.
        
        # Vectorized approach for the group:
        targets = 1.5 * group_indices.astype(np.float64)
        pos = np.searchsorted(group_indices, targets, side='left')
        
        # pos contains insertion points.
        # We need to check group_indices[pos] and group_indices[pos-1].
        # Handle boundaries: pos might be 0 or len(group_indices).
        
        # Vectorized candidate extraction
        # Candidate 1: group_indices[pos] where pos < size
        mask1 = pos < size
        cand1 = np.zeros(size, dtype=np.int64)
        cand1[mask1] = group_indices[pos[mask1]]
        
        # Candidate 2: group_indices[pos-1] where pos > 0
        mask2 = pos > 0
        cand2 = np.zeros(size, dtype=np.int64)
        cand2[mask2] = group_indices[pos[mask2] - 1]
        
        # Calculate errors for candidates
        # err = |a/b - 1.5|
        
        # For cand1
        valid1 = mask1 & (cand1 != group_indices)  # a != b
        if np.any(valid1):
            a_vals = cand1[valid1]
            b_vals = group_indices[valid1]
            errors = np.abs(a_vals / b_vals - 1.5)
            min_idx = np.argmin(errors)
            min_err = errors[min_idx]
            if min_err < best_error:
                best_error = min_err
                best_a = int(a_vals[min_idx])
                best_b = int(b_vals[min_idx])
        
        # For cand2
        valid2 = mask2 & (cand2 != group_indices)  # a != b
        if np.any(valid2):
            a_vals = cand2[valid2]
            b_vals = group_indices[valid2]
            errors = np.abs(a_vals / b_vals - 1.5)
            min_idx = np.argmin(errors)
            min_err = errors[min_idx]
            if min_err < best_error:
                best_error = min_err
                best_a = int(a_vals[min_idx])
                best_b = int(b_vals[min_idx])
        
        # Check deadline
        if time.time() - start_time > 50:
            break

    return [int(best_a), int(best_b)]
