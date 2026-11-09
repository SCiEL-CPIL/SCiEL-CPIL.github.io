import numpy as np
import time
from collections import defaultdict

def construct():
    t_start = time.time()
    deadline = t_start + 80.0
    
    N = 10_000_000
    
    # 1. Compute sigma(n) for n in [1, N] using sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    if time.time() > deadline:
        return [1, 1]

    # 2. Group indices by sigma value
    sigma_to_indices = defaultdict(list)
    for n in range(1, N + 1):
        sigma_to_indices[int(sigma[n])].append(n)
        
    if time.time() > deadline:
        return [1, 1]

    # 3. For each group with at least 2 elements, find pair minimizing |a/b - 3/2|
    best_err = float('inf')
    best_a = 1
    best_b = 1
    
    for s_val, indices in sigma_to_indices.items():
        if time.time() > deadline - 2.0:
            break
        if len(indices) < 2:
            continue
            
        # Sort indices (they should already be sorted since we iterate n in order)
        arr = np.array(indices, dtype=np.int64)
        
        # For each b in arr, find a in arr closest to 1.5*b
        # Use searchsorted for efficiency
        # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
        # So for each b, we want a closest to 1.5*b
        
        # Vectorized approach: for each b, target = 1.5*b
        # Find position in sorted arr
        targets = 1.5 * arr.astype(np.float64)
        positions = np.searchsorted(arr, targets)
        
        # Check position-1 and position for each
        # Handle boundaries
        pos_minus = np.clip(positions - 1, 0, len(arr) - 1)
        pos_plus = np.clip(positions, 0, len(arr) - 1)
        
        a_candidates_minus = arr[pos_minus]
        a_candidates_plus = arr[pos_plus]
        
        # Compute errors for both candidates
        # err = |a/b - 1.5|
        # But we need a != b for valid pair (error > 0)
        
        # For minus candidates
        b_vals = arr
        a_minus = a_candidates_minus
        a_plus = a_candidates_plus
        
        # Compute |a/b - 1.5| for each
        # Avoid division by zero (b >= 1 always)
        err_minus = np.abs(a_minus.astype(np.float64) / b_vals.astype(np.float64) - 1.5)
        err_plus = np.abs(a_plus.astype(np.float64) / b_vals.astype(np.float64) - 1.5)
        
        # Mask out cases where a == b
        mask_minus = a_minus != b_vals
        mask_plus = a_plus != b_vals
        
        err_minus = np.where(mask_minus, err_minus, np.inf)
        err_plus = np.where(mask_plus, err_plus, np.inf)
        
        # Find minimum
        min_idx_minus = np.argmin(err_minus)
        min_err_minus = err_minus[min_idx_minus]
        
        min_idx_plus = np.argmin(err_plus)
        min_err_plus = err_plus[min_idx_plus]
        
        if min_err_minus < best_err:
            best_err = min_err_minus
            best_a = int(a_minus[min_idx_minus])
            best_b = int(b_vals[min_idx_minus])
            
        if min_err_plus < best_err:
            best_err = min_err_plus
            best_a = int(a_plus[min_idx_plus])
            best_b = int(b_vals[min_idx_plus])
            
    if best_err == float('inf'):
        return [1, 1]
        
    return [int(best_a), int(best_b)]
