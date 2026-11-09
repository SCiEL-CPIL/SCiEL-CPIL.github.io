"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from math import gcd
import time

def construct():
    N = 512
    size = 511  # numbers 1..511
    
    # Weights: 1 / (512 - a) for a in 1..511
    # a=1 -> 1/511, a=511 -> 1/1
    weights = np.array([1.0 / (N - (i + 1)) for i in range(size)])
    
    # Precompute conflict matrix
    # conflict[i][j] is True if gcd(i+1, j+1) > 1
    conflict = np.zeros((size, size), dtype=bool)
    for i in range(size):
        a = i + 1
        for j in range(i + 1, size):
            b = j + 1
            if gcd(a, b) > 1:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # Greedy initialization: pick from 511 down to 1 (highest weight first)
    chosen_mask = np.zeros(size, dtype=bool)
    # Iterate in order of decreasing weight (increasing index a)
    for i in range(size - 1, -1, -1):
        # Check if i conflicts with any already chosen
        if not np.any(conflict[i] & chosen_mask):
            chosen_mask[i] = True
            
    # Local Search: Vectorized swaps
    start_time = time.time()
    improved = True
    # Limit iterations to ensure time safety, though vectorization is fast
    max_iter = 10000 
    
    while improved and (time.time() - start_time) < 110 and max_iter > 0:
        improved = False
        max_iter -= 1
        chosen_indices = np.where(chosen_mask)[0]
        
        # Try swapping each chosen element with a better one
        for idx_a in chosen_indices:
            a_val = idx_a + 1
            weight_a = weights[idx_a]
            
            # Temporarily remove idx_a
            temp_mask = chosen_mask.copy()
            temp_mask[idx_a] = False
            
            # Find all candidates b that are NOT chosen and NOT conflicting with any in temp_mask
            # blocked[j] is True if j conflicts with ANY k in temp_mask
            if np.any(temp_mask):
                blocked = np.any(conflict[:, temp_mask], axis=1)
            else:
                blocked = np.zeros(size, dtype=bool)
                
            valid = ~blocked & ~temp_mask
            
            if not np.any(valid):
                continue
                
            # Calculate gain for each valid candidate
            # gain[b] = weight[b] - weight_a
            gains = weights - weight_a
            gains[~valid] = -np.inf
            
            best_idx = np.argmax(gains)
            best_gain = gains[best_idx]
            
            if best_gain > 1e-12:
                # Perform swap
                chosen_mask[idx_a] = False
                chosen_mask[best_idx] = True
                improved = True
                break # Restart loop after one swap to refresh indices
                
    return [int(i + 1) for i in np.where(chosen_mask)[0]]
# EVOLVE-BLOCK-END
