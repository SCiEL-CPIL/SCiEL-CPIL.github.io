import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Filter to feasible items
    # The problem statement says "remaining (feasible) arrays", implying 
    # that inputs might already be filtered? 
    # "index into the remaining (feasible) arrays"
    # Let's assume the inputs contain only feasible items or we must check.
    # The prompt says "remaining_weights, remaining_values" are the arrays.
    # Usually in such loops, the caller removes the picked item.
    # If an item is infeasible, it shouldn't be picked.
    # Let's check feasibility explicitly to be safe.
    
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    
    # If only one feasible item, pick it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
        
    # Pre-compute density and sort indices by density descending
    # We need to handle division by zero, but weights should be > 0.
    densities = np.zeros(n)
    # Avoid division by zero
    safe_weights = np.where(remaining_weights > 0, remaining_weights, 1e-9)
    densities = remaining_values / safe_weights
    
    # Sort indices by density, highest first
    # We only care about feasible items for the "rest" estimation too?
    # Yes, infeasible items can't be picked in the future either if capacity decreases.
    # So we can just sort all, but only consider feasible ones in the loop.
    sorted_indices = np.argsort(densities)[::-1]
    
    best_score = -np.inf
    best_idx = -1
    
    # For each feasible candidate, estimate total value
    for i in feasible_indices:
        w_i = remaining_weights[i]
        v_i = remaining_values[i]
        
        rem_cap = remaining_capacity - w_i
        
        # Estimate greedy fill of the rest
        # Iterate through sorted items, skip i, add if fits
        est_val = v_i
        
        # Optimization: If rem_cap is small, we might break early?
        # No, we need to sum all that fit.
        
        for j in sorted_indices:
            if j == i:
                continue
            # Only consider feasible items in the "rest" set
            # Since rem_cap decreases, if an item was infeasible before (w > C), 
            # it is definitely infeasible now (w > C - w_i).
            # So we can just check if it fits in rem_cap.
            if remaining_weights[j] <= rem_cap:
                est_val += remaining_values[j]
                rem_cap -= remaining_weights[j]
                if rem_cap <= 0:
                    break
        
        if est_val > best_score:
            best_score = est_val
            best_idx = i
            
    return int(best_idx)
