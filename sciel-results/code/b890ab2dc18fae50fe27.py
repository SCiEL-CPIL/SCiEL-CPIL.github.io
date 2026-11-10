"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Safety margin before 120s kill


def _fractional_bound(weights, values, capacity, start_idx, sorted_indices):
    """
    Calculate the fractional knapsack upper bound starting from a specific index
    in a pre-sorted list of items (by density).
    """
    if capacity <= 0:
        return 0.0
    
    total_val = 0.0
    remaining_cap = capacity
    
    # Iterate through items in density order
    for i in range(start_idx, len(sorted_indices)):
        idx = sorted_indices[i]
        w = weights[idx]
        v = values[idx]
        
        if w <= remaining_cap:
            total_val += v
            remaining_cap -= w
        else:
            # Fractional part
            if w > 0:
                total_val += (remaining_cap / w) * v
            break
            
    return total_val


def _branch_and_bound(weights, values, capacity, time_limit_check=None, time_limit_val=0):
    """
    Solve the 0/1 knapsack problem exactly using Branch and Bound.
    Returns the maximum value achievable.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    # Sort items by density (value/weight) descending
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = values / weights
    # Handle zero weights (infinite density)
    densities = np.where(densities == np.inf, np.finfo(float).max, densities)
    sorted_indices = np.argsort(-densities, kind='stable')
    
    # Pre-extract sorted weights and values for faster access in the loop
    sorted_weights = weights[sorted_indices]
    sorted_values = values[sorted_indices]
    
    best_value = 0.0
    
    # Initial bound check
    initial_bound = _fractional_bound(sorted_weights, sorted_values, capacity, 0, np.arange(n))
    if initial_bound <= best_value:
        return best_value

    # Stack for DFS: (next_item_index, current_weight, current_value)
    # We use a list as a stack.
    # To explore the 'include' branch first (which is often better for finding good solutions early),
    # we push 'exclude' first, then 'include'.
    stack = [(0, 0.0, 0.0)]
    
    while stack:
        # Time check
        if time_limit_check is not None and time.time() > time_limit_check:
            return best_value
            
        i, curr_w, curr_v = stack.pop()
        
        if curr_w > capacity:
            continue
            
        # Update best value
        if curr_v > best_value:
            best_value = curr_v
            
        # Check if we've processed all items
        if i >= n:
            continue
            
        # Calculate upper bound for this node
        # The bound assumes we can take fractional items from i onwards
        ub = curr_v + _fractional_bound(sorted_weights, sorted_values, capacity - curr_w, i, np.arange(i, n))
        
        if ub <= best_value:
            continue
            
        # Branch 1: Exclude item i
        stack.append((i + 1, curr_w, curr_v))
        
        # Branch 2: Include item i
        new_w = curr_w + sorted_weights[i]
        if new_w <= capacity:
            new_v = curr_v + sorted_values[i]
            stack.append((i + 1, new_w, new_v))
            
    return best_value


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select the next item using Branch and Bound lookahead.
    Returns index into remaining arrays, or -1 to stop.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return -1
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Filter feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = remaining_weights[feasible_indices]
    feasible_values = remaining_values[feasible_indices]
    
    if len(feasible_indices) == 0:
        return -1
        
    # Calculate densities for feasible items
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = feasible_values / feasible_weights
    densities = np.where(densities == np.inf, np.finfo(float).max, densities)
    sorted_local = np.argsort(-densities, kind='stable')
    
    # Determine how many candidates to evaluate
    # We want to balance accuracy and speed. 
    # If we have many items, B&B on the sub-problem is the bottleneck.
    # Let's evaluate top K candidates.
    K = min(15, len(feasible_indices))
    top_k_local = sorted_local[:K]
    
    best_total_value = -1.0
    best_global_idx = -1
    
    # We need to be careful with time. 
    # If we run out of time, we should return the best greedy candidate found so far.
    # Let's set a sub-time-limit for the B&B calls to prevent any single call from taking too long.
    # However, since we don't know how many calls we'll make, it's safer to check global time.
    
    for local_idx in top_k_local:
        # Check time before starting a new candidate evaluation
        if time.time() - _START_TIME > _TIME_LIMIT - 2:
            break
            
        global_idx = feasible_indices[local_idx]
        w_pick = remaining_weights[global_idx]
        v_pick = remaining_values[global_idx]
        
        rem_cap = remaining_capacity - w_pick
        if rem_cap < 0:
            continue
            
        # Prepare arrays for the sub-problem (excluding the picked item)
        # We create a mask of items that are feasible and not the picked one
        sub_mask = feasible_mask.copy()
        sub_mask[global_idx] = False
        
        # Get indices of sub-problem items
        sub_indices = np.where(sub_mask)[0]
        
        if len(sub_indices) == 0:
            total_val = v_pick
        else:
            sub_weights = remaining_weights[sub_indices]
            sub_values = remaining_values[sub_indices]
            
            # Solve the sub-problem using B&B
            # Pass a time check to the B&B function
            sub_val = _branch_and_bound(
                sub_weights, 
                sub_values, 
                rem_cap,
                time_limit_check=_TIME_LIMIT - 1,
                time_limit_val=_TIME_LIMIT
            )
            total_val = v_pick + sub_val
                
        if total_val > best_total_value:
            best_total_value = total_val
            best_global_idx = global_idx
            
    if best_global_idx == -1:
        # Fallback to greedy if B&B didn't yield a result (e.g., due to time)
        # Pick the highest density feasible item
        best_local = sorted_local[0]
        return int(feasible_indices[best_local])
        
    return int(best_global_idx)

# EVOLVE-BLOCK-END
