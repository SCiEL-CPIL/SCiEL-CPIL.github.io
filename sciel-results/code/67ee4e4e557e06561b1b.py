"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.monotonic()
_TIME_LIMIT = 110.0  # Safety margin for 120s limit

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Selects the next item for a 0/1 Knapsack problem using a heuristic that
    evaluates the immediate value plus the estimated value of the remaining
    capacity filled via Linear Programming relaxation of the remaining items.
    """
    # Time guard
    if time.monotonic() - _START_TIME > _TIME_LIMIT:
        return -1

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Convert to numpy arrays if not already
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)

    # Find feasible items
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    
    # Pre-compute density for sorting
    # Avoid division by zero, though weights should be > 0 in knapsack
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = values / weights
    
    # Sort indices by density descending
    # We only care about the order of all items, we will skip the candidate in the inner loop
    sorted_indices = np.argsort(densities)[::-1]
    
    best_score = -1.0
    best_idx = -1
    
    # Evaluate each feasible candidate
    for i in feasible_indices:
        w_i = weights[i]
        v_i = values[i]
        rem_cap = remaining_capacity - w_i
        
        # Calculate LP relaxation value for remaining items in rem_cap
        lp_val = 0.0
        cur_cap = rem_cap
        
        # Iterate through items sorted by density
        for j in sorted_indices:
            if j == i:
                continue
            w_j = weights[j]
            v_j = values[j]
            
            if w_j <= cur_cap:
                lp_val += v_j
                cur_cap -= w_j
                if cur_cap <= 0:
                    break
            else:
                # Fractional part
                if w_j > 0:
                    lp_val += v_j * (cur_cap / w_j)
                cur_cap = 0
                break
        
        total_score = v_i + lp_val
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
            
    return int(best_idx)
# EVOLVE-BLOCK-END
