import time
import numpy as np
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    n = 30
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)
        
    weights = [L // (i + 1) for i in range(n)]
    
    half = 15
    left_weights = weights[:half]
    right_weights = weights[half:]
    
    # Generate left sums with path tracking for reconstruction
    left_sums = [0]
    left_paths = [()]
    for w in left_weights:
        new_sums = []
        new_paths = []
        for s, p in zip(left_sums, left_paths):
            new_sums.append(s)
            new_paths.append(p + (0,))
            new_sums.append(s + w)
            new_paths.append(p + (1,))
            new_sums.append(s - w)
            new_paths.append(p + (-1,))
        left_sums = new_sums
        left_paths = new_paths
        
    if time.time() > deadline:
        return [0]*28 + [1, -1]

    # Generate right sums with path tracking
    right_sums = [0]
    right_paths = [()]
    for w in right_weights:
        new_sums = []
        new_paths = []
        for s, p in zip(right_sums, right_paths):
            new_sums.append(s)
            new_paths.append(p + (0,))
            new_sums.append(s + w)
            new_paths.append(p + (1,))
            new_sums.append(s - w)
            new_paths.append(p + (-1,))
        right_sums = new_sums
        right_paths = new_paths

    if time.time() > deadline:
        return [0]*28 + [1, -1]

    left_arr = np.array(left_sums, dtype=np.int64)
    right_arr = np.array(right_sums, dtype=np.int64)
    
    # Sort left sums and keep track of original indices
    left_order = np.argsort(left_arr)
    left_sorted = left_arr[left_order]
    
    # For each right sum, find the left sum that minimizes |left + right| != 0
    # We want to minimize |left + right| subject to left + right != 0
    
    best_val = None
    best_left_idx = 0
    best_right_idx = 0
    
    # Use searchsorted to find insertion points
    # For target = -right_sum, we want left_sum closest to target
    # But left_sum + right_sum != 0 means left_sum != -right_sum
    
    # Vectorized approach: for each right value, find the best left value
    targets = -right_arr
    
    # Find insertion points in sorted left array
    insert_pos = np.searchsorted(left_sorted, targets)
    
    # Check positions: insert_pos-1, insert_pos, insert_pos+1
    # But we need left_sorted[pos] != target (i.e., left_sorted[pos] + right_arr != 0)
    
    # Create candidate positions
    n_right = len(right_arr)
    candidates = np.empty((3, n_right), dtype=np.int64)
    candidates[0] = np.clip(insert_pos - 1, 0, len(left_sorted) - 1)
    candidates[1] = np.clip(insert_pos, 0, len(left_sorted) - 1)
    candidates[2] = np.clip(insert_pos + 1, 0, len(left_sorted) - 1)
    
    # Get the actual left values for each candidate
    candidate_left_vals = left_sorted[candidates]
    
    # Compute the sums: candidate_left_val + right_val
    # candidate_left_vals shape: (3, n_right)
    # right_arr shape: (n_right,) -> broadcast to (3, n_right)
    sums = candidate_left_vals + right_arr[np.newaxis, :]
    
    # Mask out zeros
    nonzero_mask = sums != 0
    
    # For positions where the sum is zero, set absolute value to infinity
    abs_sums = np.where(nonzero_mask, np.abs(sums), np.int64(2**62))
    
    # Find the minimum across all 3 candidates for each right
    min_abs_per_right = np.min(abs_sums, axis=0)
    
    # Find the overall minimum
    global_min_idx = np.argmin(min_abs_per_right)
    best_right_idx = global_min_idx
    best_val = min_abs_per_right[global_min_idx]
    
    # Find which candidate (0, 1, or 2) gave the minimum for this right
    candidate_idx = np.argmin(abs_sums[:, global_min_idx])
    best_left_sorted_idx = candidates[candidate_idx, global_min_idx]
    
    # Map back to original left index
    best_left_idx = left_order[best_left_sorted_idx]
    
    if time.time() > deadline:
        return [0]*28 + [1, -1]
    
    # Reconstruct coefficients from paths
    left_coeffs = list(left_paths[best_left_idx])
    right_coeffs = list(right_paths[best_right_idx])
    
    if len(left_coeffs) != half or len(right_coeffs) != (n - half):
        return [0]*28 + [1, -1]
    
    return left_coeffs + right_coeffs
