import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Ensure inputs are numpy arrays for performance if they aren't already
    # The prompt implies they are arrays or list-like.
    # np.asarray is safe.
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    
    # Calculate densities
    # Handle zero weights gracefully
    densities = np.zeros(n, dtype=float)
    non_zero_mask = weights > 0
    densities[non_zero_mask] = values[non_zero_mask] / weights[non_zero_mask]
    # Items with zero weight have infinite density (or we can just put them first)
    # If weight is 0, it always fits.
    # Let's assign a very high density to zero weight items if value > 0
    zero_weight_mask = weights == 0
    densities[zero_weight_mask] = np.inf
    
    # Sort indices by density descending
    sorted_indices = np.argsort(densities)[::-1]
    
    best_val = -1.0
    best_idx = -1
    
    # For each candidate first item
    for i in range(n):
        w_i = weights[i]
        if w_i > remaining_capacity:
            continue
            
        # Simulate greedy fill for the rest
        sim_cap = remaining_capacity - w_i
        sim_val = values[i]
        
        for j in sorted_indices:
            if j == i:
                continue
            w_j = weights[j]
            if w_j <= sim_cap:
                sim_cap -= w_j
                sim_val += values[j]
            # else: skip
            
        if sim_val > best_val:
            best_val = sim_val
            best_idx = i
            
    return int(best_idx)
