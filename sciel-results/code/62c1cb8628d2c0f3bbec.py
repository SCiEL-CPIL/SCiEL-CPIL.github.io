import numpy as np
import time

def _solve_small_knapsack(weights, values, capacity):
    """Exact solver for small n using bitmask."""
    n = len(weights)
    if n == 0:
        return 0.0
    best_val = 0.0
    # Iterate through all subsets
    for mask in range(1 << n):
        w = 0.0
        v = 0.0
        # Early termination if weight exceeds capacity
        for i in range(n):
            if mask & (1 << i):
                w += weights[i]
                if w > capacity:
                    break
                v += values[i]
        if w <= capacity:
            if v > best_val:
                best_val = v
    return best_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Check for feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])

    start_time = time.time()
    deadline = start_time + 5.0 # 5 seconds per step is very safe
    
    if n <= 15:
        best_val = -np.inf
        best_idx = -1
        for i in feasible_indices:
            if time.time() > deadline:
                break
            wi = remaining_weights[i]
            vi = remaining_values[i]
            rem_cap = remaining_capacity - wi
            
            # Prepare remaining arrays
            rem_weights = np.delete(remaining_weights, i)
            rem_values = np.delete(remaining_values, i)
            
            exact_val = _solve_small_knapsack(rem_weights, rem_values, rem_cap)
            total_val = vi + exact_val
            
            if total_val > best_val:
                best_val = total_val
                best_idx = i
        return int(best_idx)
    else:
        # 1-step lookahead with greedy remainder
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = np.where(remaining_weights > 0, remaining_values / remaining_weights, 0)
        sorted_indices = np.argsort(-densities)
        
        best_val = -np.inf
        best_idx = -1
        
        for i in feasible_indices:
            if time.time() > deadline:
                break
                
            wi = remaining_weights[i]
            vi = remaining_values[i]
            
            rem_cap = remaining_capacity - wi
            total_val = vi
            
            for j in sorted_indices:
                if j == i:
                    continue
                if remaining_weights[j] <= rem_cap:
                    total_val += remaining_values[j]
                    rem_cap -= remaining_weights[j]
                    if rem_cap <= 1e-9:
                        break
            
            if total_val > best_val:
                best_val = total_val
                best_idx = i
                
        return int(best_idx)
