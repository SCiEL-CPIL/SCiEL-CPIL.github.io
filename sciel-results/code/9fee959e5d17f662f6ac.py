"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    c = float(remaining_capacity)
    
    # Check if any item fits
    feasible_mask = w <= c
    if not np.any(feasible_mask):
        return -1
        
    # Get indices of feasible items
    feasible_indices = np.where(feasible_mask)[0]
    
    # If only one feasible item, return it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
        
    # Calculate densities for feasible items
    # Avoid division by zero, though weights are usually > 0
    w_f = w[feasible_indices]
    v_f = v[feasible_indices]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v_f / w_f
        # Replace inf with a large number
        densities = np.where(np.isinf(densities), 1e18, densities)
        densities = np.where(np.isnan(densities), 0, densities)
        
    # Sort feasible indices by density descending
    # We need to map back to original indices
    sorted_order = np.argsort(-densities)
    sorted_feasible_indices = feasible_indices[sorted_order]
    
    best_value = -1
    best_index = -1
    
    # 1-Step Lookahead
    # For each candidate item, simulate the rest of the greedy process
    # We iterate through the sorted list. 
    # Note: The "greedy" simulation assumes we pick items in density order 
    # if they fit, skipping those that don't.
    
    # To speed up, we can pre-compute nothing, just do the O(N^2) loop.
    # N is max 100. 100 * 100 = 10,000 iterations. Very fast.
    
    for i in range(len(sorted_feasible_indices)):
        candidate_idx = sorted_feasible_indices[i]
        
        # Pick this candidate
        rem_cap = c - w[candidate_idx]
        total_val = v[candidate_idx]
        
        # Simulate greedy on remaining items
        # We iterate through the sorted list again
        # If we encounter the candidate, skip it.
        # If an item fits, take it.
        # If it doesn't fit, skip it (standard greedy behavior for 0/1 with sorted list).
        
        for j in range(len(sorted_feasible_indices)):
            if j == i:
                continue
            
            item_idx = sorted_feasible_indices[j]
            item_w = w[item_idx]
            
            if item_w <= rem_cap:
                total_val += v[item_idx]
                rem_cap -= item_w
            # else: skip
            
        if total_val > best_value:
            best_value = total_val
            best_index = int(candidate_idx)
            
    return best_index
# EVOLVE-BLOCK-END
