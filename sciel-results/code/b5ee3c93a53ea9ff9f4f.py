import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 140.0

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _check_time():
    if _START_TIME is None:
        return True
    return (time.time() - _START_TIME) < _TIME_LIMIT

def _greedy_fallback(cap, weights, values):
    """Fallback greedy if DP is too slow or fails."""
    if len(weights) == 0 or cap <= 0:
        return -1
    # Filter feasible
    mask = weights <= cap
    if not np.any(mask):
        return -1
    f_weights = weights[mask]
    f_values = values[mask]
    f_indices = np.where(mask)[0]
    
    density = f_values / np.where(f_weights > 0, f_weights, 1e-9)
    order = np.argsort(-density)
    cap_rem = cap
    for idx in order:
        if f_weights[idx] <= cap_rem:
            return int(f_indices[idx])
    return -1

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    _init_time()
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    cap = remaining_capacity
    
    # Filter out infeasible items
    feasible_mask = weights <= cap
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = weights[feasible_mask]
    feasible_values = values[feasible_mask]
    n_feasible = len(feasible_indices)
    
    if n_feasible == 1:
        return int(feasible_indices[0])
    
    # Scale to integers
    scale = 100
    cap_int = int(round(cap * scale))
    weights_int = np.round(feasible_weights * scale).astype(int)
    
    # Handle zero/negative weights
    zero_weight_indices = []
    non_zero_indices = []
    for i in range(n_feasible):
        if weights_int[i] <= 0:
            if feasible_values[i] > 0:
                zero_weight_indices.append(i)
        else:
            non_zero_indices.append(i)
            
    if not non_zero_indices:
        # Only zero weight items
        if zero_weight_indices:
            # Pick highest value
            best_idx = zero_weight_indices[np.argmax(feasible_values[zero_weight_indices])]
            return int(feasible_indices[best_idx])
        else:
            return -1
            
    if cap_int <= 0:
        # No capacity, can only pick zero weight items
        if zero_weight_indices:
            best_idx = zero_weight_indices[np.argmax(feasible_values[zero_weight_indices])]
            return int(feasible_indices[best_idx])
        return -1
        
    # DP for non-zero items
    # Map non-zero indices to a new array for DP
    nz_weights = weights_int[non_zero_indices]
    nz_values = feasible_values[non_zero_indices]
    n_nz = len(nz_weights)
    
    # Fallback if too large
    if cap_int > 5000 or n_nz * cap_int > 1000000:
        return _greedy_fallback(cap, feasible_weights, feasible_values)
        
    dp = np.zeros(cap_int + 1, dtype=np.float64)
    take = np.zeros((n_nz, cap_int + 1), dtype=bool)
    
    for i in range(n_nz):
        if not _check_time():
            return _greedy_fallback(cap, feasible_weights, feasible_values)
        w_i = nz_weights[i]
        v_i = nz_values[i]
        if w_i > cap_int:
            continue
        for c in range(cap_int, w_i - 1, -1):
            if dp[c - w_i] + v_i > dp[c]:
                dp[c] = dp[c - w_i] + v_i
                take[i][c] = True
    
    # Reconstruct
    c = cap_int
    optimal_nz_indices = []
    for i in range(n_nz - 1, -1, -1):
        if take[i][c]:
            optimal_nz_indices.append(i)
            c -= nz_weights[i]
            
    # Map back to feasible_indices
    optimal_set_indices = []
    for idx in optimal_nz_indices:
        optimal_set_indices.append(non_zero_indices[idx])
        
    # Add zero weight items
    optimal_set_indices.extend(zero_weight_indices)
    
    if not optimal_set_indices:
        return _greedy_fallback(cap, feasible_weights, feasible_values)
        
    # Pick the best item from the optimal set
    max_val = -1
    best_idx = -1
    for idx in optimal_set_indices:
        if feasible_values[idx] > max_val:
            max_val = feasible_values[idx]
            best_idx = idx
            
    return int(feasible_indices[best_idx])
