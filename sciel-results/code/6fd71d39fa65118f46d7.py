import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: pick the next item to add.
    Uses a 1-step look-ahead with greedy completion.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    n = len(remaining_weights)
    if n == 0:
        return -1

    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    cap = float(remaining_capacity)

    # Find feasible items
    feasible_mask = weights <= cap
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    f_weights = weights[feasible_mask]
    f_values = values[feasible_mask]
    
    # If only one feasible item, pick it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])

    # Calculate densities for sorting
    # Handle zero weights
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(f_weights > 0, f_values / f_weights, np.inf)
    
    # If any item has weight 0 and value > 0, it's "free". 
    # We should probably pick the highest value free item first? 
    # Or just let the look-ahead handle it. 
    # If w=0, it doesn't reduce capacity. 
    # The look-ahead simulation will naturally include them if they fit (always fit).
    # However, sorting by density with inf values puts them first.
    
    # Sort feasible items by density descending
    # We need to keep track of original indices
    # Create a list of (density, original_index)
    # For stability, if densities are equal, maybe sort by value desc?
    sort_order = np.argsort(-densities)
    sorted_feasible_indices = feasible_indices[sort_order]
    
    # We will evaluate each feasible item as the "next pick"
    # and simulate the rest using the sorted order
    
    best_total_val = -1
    best_pick_idx = -1
    
    # Optimization: Only consider top K candidates for the "next pick"?
    # The best first pick is often near the top of density, but not always.
    # Given N=100, evaluating all is fast enough.
    
    for i in range(len(feasible_indices)):
        if time.time() > deadline:
            # Fallback to simple density if running out of time
            # Pick the first one in sorted order (highest density)
            return int(sorted_feasible_indices[0])
            
        cand_idx = feasible_indices[i]
        cand_w = f_weights[i]
        cand_v = f_values[i]
        
        rem_cap = cap - cand_w
        total_val = cand_v
        
        # Simulate greedy fill with remaining items
        # We iterate through sorted_feasible_indices
        # We must skip cand_idx
        for next_idx in sorted_feasible_indices:
            if next_idx == cand_idx:
                continue
            # Get weight and value for next_idx
            # We can use the original arrays for O(1) access
            w_next = weights[next_idx]
            if w_next <= rem_cap:
                rem_cap -= w_next
                total_val += values[next_idx]
                
        if total_val > best_total_val:
            best_total_val = total_val
            best_pick_idx = cand_idx

    if best_pick_idx == -1:
        # Fallback
        return int(sorted_feasible_indices[0])
        
    return int(best_pick_idx)
