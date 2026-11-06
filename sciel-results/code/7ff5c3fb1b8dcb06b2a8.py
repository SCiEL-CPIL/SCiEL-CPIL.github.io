"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def _greedy_value_estimate(weights, values, capacity):
    """
    Estimate the maximum value achievable with the given items and capacity
    using a simple greedy density heuristic.
    """
    if capacity <= 0 or len(weights) == 0:
        return 0.0
    
    # Filter items that individually fit
    mask = weights <= capacity
    if not np.any(mask):
        return 0.0
    
    w = weights[mask]
    v = values[mask]
    
    # Calculate density
    density = v / w
    
    # Sort by density descending
    sorted_indices = np.argsort(-density)
    
    total_value = 0.0
    remaining_cap = capacity
    
    for idx in sorted_indices:
        if w[idx] <= remaining_cap:
            total_value += v[idx]
            remaining_cap -= w[idx]
        else:
            # Since we are sorted by density, if this doesn't fit, 
            # we can't just skip it and take a lower density one that might fit?
            # Actually, in 0/1 knapsack greedy, we just skip and continue.
            # But for a fast estimate, we can break if we assume weights are sorted? No.
            # Just continue.
            pass
            
    return total_value

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if len(remaining_weights) == 0:
        return -1
    
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    n = len(weights)
    
    # Find feasible items
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    
    best_score = -1.0
    best_idx = -1
    
    # For each feasible item, calculate lookahead score
    for idx in feasible_indices:
        # Value if we take this item
        val = values[idx]
        
        # Capacity remaining after taking this item
        cap_after = remaining_capacity - weights[idx]
        
        if cap_after <= 0:
            future_val = 0.0
        else:
            # Create arrays for remaining items (excluding idx)
            # We can use a mask to exclude idx
            remaining_mask = np.ones(n, dtype=bool)
            remaining_mask[idx] = False
            
            # Only consider items that are still feasible for the new capacity?
            # The _greedy_value_estimate function handles filtering internally.
            w_rem = weights[remaining_mask]
            v_rem = values[remaining_mask]
            
            future_val = _greedy_value_estimate(w_rem, v_rem, cap_after)
        
        total_score = val + future_val
        
        if total_score > best_score:
            best_score = total_score
            best_idx = idx
            
    return int(best_idx)
# EVOLVE-BLOCK-END
