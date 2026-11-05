import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Check for zero weight items with positive value
    zero_weight_mask = (remaining_weights == 0) & (remaining_values > 0)
    if np.any(zero_weight_mask):
        # Pick the first one (or any)
        return int(np.where(zero_weight_mask)[0][0])

    # Filter feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = remaining_weights[feasible_mask]
    feasible_values = remaining_values[feasible_mask]
    
    # Calculate densities
    densities = feasible_values / feasible_weights
    
    # Sort indices by density descending
    sorted_order = np.argsort(-densities)
    
    best_score = -1
    best_idx = -1
    start_time = time.time()
    deadline = start_time + 1.0 # Safety margin
    
    for k in range(len(feasible_indices)):
        if time.time() > deadline:
            # Fallback to density greedy
            best_idx = feasible_indices[sorted_order[0]]
            break
            
        idx_in_feasible = feasible_indices[sorted_order[k]]
        w = feasible_weights[sorted_order[k]]
        v = feasible_values[sorted_order[k]]
        
        # Simulate picking this item
        rem_cap = remaining_capacity - w
        sim_val = v
        
        # Greedy fill the rest
        # Iterate through sorted items
        for j in range(len(feasible_indices)):
            if j == sorted_order[k]:
                continue
            w_j = feasible_weights[sorted_order[j]]
            if w_j <= rem_cap:
                sim_val += feasible_values[sorted_order[j]]
                rem_cap -= w_j
                if rem_cap <= 0:
                    break
        
        if sim_val > best_score:
            best_score = sim_val
            best_idx = idx_in_feasible

    if best_idx == -1:
        best_idx = feasible_indices[sorted_order[0]]
        
    return int(best_idx)
