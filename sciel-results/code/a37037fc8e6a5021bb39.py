import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Direct heuristic for 0/1 knapsack: pick the item that, when combined with a
    greedy density-based fill of the remaining capacity, yields the highest total value.
    """
    start_time = time.time()
    deadline = 145.0  # 5-second safety margin before 150s kill
    
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    
    n = len(weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Quick check: if no item fits, stop
    if np.all(weights > remaining_capacity):
        return -1
    
    # Filter to only feasible items (weight <= remaining_capacity)
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    f_weights = weights[feasible_mask]
    f_values = values[feasible_mask]
    f_n = len(f_weights)
    
    if f_n == 0:
        return -1
    
    # Compute densities (value / weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(f_weights > 0, f_values / f_weights, np.inf)
    
    # Sort by density descending; tie-break by higher value, then lower weight
    sort_keys = np.lexsort((f_weights, -f_values, -densities))
    sorted_f_indices = feasible_indices[sort_keys]
    sorted_weights = f_weights[sort_keys]
    sorted_values = f_values[sort_keys]
    sorted_densities = densities[sort_keys]
    
    # For a small number of feasible items, do a full lookahead
    # For larger, limit candidates to top K
    K = min(f_n, 30)
    
    best_total_value = -1
    best_global_idx = -1
    
    for k in range(K):
        # Check deadline
        if time.time() - start_time > deadline - 5.0:
            break
        
        i = k
        if sorted_weights[i] > remaining_capacity:
            continue
        
        # Simulate picking item i
        curr_cap = remaining_capacity - sorted_weights[i]
        total_val = sorted_values[i]
        
        # Calculate Fractional Knapsack Value for remaining items
        # This serves as a smooth surrogate for the optimal integer fill
        frac_val = 0.0
        for j in range(f_n):
            if j == i:
                continue
            if curr_cap <= 0:
                break
            if sorted_weights[j] <= curr_cap:
                curr_cap -= sorted_weights[j]
                frac_val += sorted_values[j]
            else:
                # Take fraction of this item
                if sorted_weights[j] > 0:
                    frac_val += sorted_values[j] * (curr_cap / sorted_weights[j])
                curr_cap = 0
                break
        
        total_score = total_val + frac_val
        
        if total_score > best_total_value:
            best_total_value = total_score
            best_global_idx = int(sorted_f_indices[i])
    
    if best_global_idx == -1:
        return -1
    
    return best_global_idx
