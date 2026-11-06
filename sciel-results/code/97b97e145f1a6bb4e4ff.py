import numpy as np
import time

_START = time.monotonic()
_TIME_LIMIT = 110.0 # Safety margin

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Check time
    if time.monotonic() - _START > _TIME_LIMIT:
        return -1 # Or a fast fallback? -1 is safer if we can't compute. 
        # Actually, if we can't compute, we should probably return -1 to stop.
        # But if we have time, we compute.

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    weights = np.asarray(remaining_weights)
    values = np.asarray(remaining_values)
    
    # Feasible items
    feasible_mask = weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1

    # Pre-sort all items by density for the inner loop
    # We need to consider all items for the "completion" part, 
    # but we only pick from feasible ones as the *first* item.
    # The completion can use any remaining item (feasible or not? No, it must fit in remaining capacity).
    # So in the inner loop, we check if item fits in remaining capacity.
    
    # Calculate density for all items
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = values / weights
    # Handle zero weights (shouldn't happen in valid knapsack, but safe)
    densities[weights == 0] = np.inf
    
    # Sort indices by density descending
    sorted_indices = np.argsort(-densities)
    
    best_value = -1
    best_idx = -1
    
    for i in feasible_indices:
        # Try picking i
        cur_val = values[i]
        rem_cap = remaining_capacity - weights[i]
        
        # Greedily fill rem_cap with other items
        for j in sorted_indices:
            if j == i:
                continue
            if weights[j] <= rem_cap:
                cur_val += values[j]
                rem_cap -= weights[j]
            # Note: We don't break here because a lighter item later might fit?
            # No, they are sorted by density, not weight.
            # A lower density item might have smaller weight and fit.
            # So we must check all.
            
        if cur_val > best_value:
            best_value = cur_val
            best_idx = i
            
    return int(best_idx)
