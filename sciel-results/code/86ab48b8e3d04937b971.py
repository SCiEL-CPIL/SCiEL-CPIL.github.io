import numpy as np
import time

_start_time = None
_GLOBAL_DEADLINE = 140.0  # Leave 10s buffer

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _is_time_up():
    return _get_elapsed() > _GLOBAL_DEADLINE

def _greedy_completion(weights, values, capacity, excluded_idx):
    """
    Greedy completion: given a set of items (with one excluded), 
    greedily pick items by density until capacity is full.
    Returns the total value achieved.
    """
    if capacity <= 0:
        return 0.0
    
    # Filter out excluded item
    mask = np.ones(len(weights), dtype=bool)
    mask[excluded_idx] = False
    w = weights[mask]
    v = values[mask]
    
    if len(w) == 0:
        return 0.0
    
    # Sort by density descending
    densities = v / w
    order = np.argsort(-densities)
    sorted_w = w[order]
    sorted_v = v[order]
    
    total_val = 0.0
    remaining_cap = capacity
    
    for i in range(len(sorted_w)):
        if sorted_w[i] <= remaining_cap:
            total_val += sorted_v[i]
            remaining_cap -= sorted_w[i]
        else:
            break
    
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if _is_time_up():
        return -1

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    weights = np.array(remaining_weights, dtype=float)
    values = np.array(remaining_values, dtype=float)
    
    # Handle zero weight positive value items first
    zero_weight_indices = np.where((weights == 0) & (values > 0))[0]
    if len(zero_weight_indices) > 0:
        return int(zero_weight_indices[0])
    
    # Filter feasible items (weight > 0 and weight <= capacity)
    feasible_mask = (weights <= remaining_capacity) & (weights > 0)
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    f_weights = weights[feasible_mask]
    f_values = values[feasible_mask]
    f_n = len(f_weights)
    
    if f_n == 0:
        return -1
    
    # Lookahead strategy: evaluate top-k density candidates
    # For each candidate, simulate adding it and then greedily completing
    # Choose the candidate that leads to the best total value
    
    C = remaining_capacity
    
    # Calculate densities
    densities = f_values / f_weights
    
    # Sort by density descending
    density_order = np.argsort(-densities)
    
    # Evaluate top-k candidates (k = min(10, f_n))
    k = min(10, f_n)
    
    best_candidate = -1
    best_total_value = -1.0
    
    for rank in range(k):
        idx = density_order[rank]
        
        # Simulate adding this item
        new_capacity = C - f_weights[idx]
        if new_capacity < 0:
            continue
        
        # Greedily complete with remaining items
        completion_value = _greedy_completion(f_weights, f_values, new_capacity, idx)
        
        # Total value if we pick this item
        total_value = f_values[idx] + completion_value
        
        if total_value > best_total_value:
            best_total_value = total_value
            best_candidate = idx
    
    if best_candidate == -1:
        # Fallback: pick the highest density item
        best_candidate = density_order[0]
    
    orig_idx = feasible_indices[best_candidate]
    return int(orig_idx)
