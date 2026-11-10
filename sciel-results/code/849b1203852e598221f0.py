import numpy as np
import time

def _fractional_bound(weights, values, capacity):
    """
    Calculate the fractional knapsack upper bound.
    weights, values: numpy arrays of same length.
    capacity: float.
    """
    if len(weights) == 0 or capacity <= 0:
        return 0.0
    
    # Calculate densities
    # Avoid division by zero, though weights should be > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = values / weights
    
    # Sort indices by density descending
    # np.argsort is stable? We want highest density first.
    # If densities are equal, order doesn't strictly matter for bound value, 
    # but for consistency, we can break ties by value or index.
    order = np.argsort(-densities)
    
    total_value = 0.0
    remaining_cap = capacity
    
    for idx in order:
        if remaining_cap <= 0:
            break
        w = weights[idx]
        v = values[idx]
        if w <= remaining_cap:
            total_value += v
            remaining_cap -= w
        else:
            total_value += v * (remaining_cap / w)
            remaining_cap = 0
            break
            
    return total_value

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Convert to numpy arrays if not already
    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)
    
    n = len(w)
    if n == 0 or remaining_capacity <= 0:
        return -1
        
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 10.0  # 10s per step is very safe, total 120s for 100 steps
    
    # Calculate current fractional bound
    ub_curr = _fractional_bound(w, v, remaining_capacity)
    
    # If ub_curr is 0, no value can be gained
    if ub_curr < 1e-9:
        return -1
        
    # Calculate opportunity costs (delta) for each item
    deltas = np.zeros(n, dtype=np.float64)
    
    # We will calculate UB without each item i
    # Optimization: We can reuse the sorting if we are careful, 
    # but for N=100, re-sorting is acceptable.
    
    for i in range(n):
        if time.time() > deadline:
            # Fallback to greedy density if we run out of time
            # Find index with max v/w
            with np.errstate(divide='ignore', invalid='ignore'):
                densities = v / w
            # Handle zero weights
            densities[w == 0] = np.inf
            return int(np.argmax(densities))
            
        # Create masks to exclude i
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        
        w_sub = w[mask]
        v_sub = v[mask]
        
        ub_sub = _fractional_bound(w_sub, v_sub, remaining_capacity)
        deltas[i] = ub_curr - ub_sub
        
    # Calculate scores: v_i / delta_i
    # We want to MINIMIZE this ratio.
    # If delta_i is very small, the ratio is huge, so it won't be picked.
    # This effectively prioritizes items that have a significant impact on the fractional bound.
    
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = v / deltas
        
    # If delta is 0, score is inf. np.argmin will ignore inf if there are finite values.
    # If all are inf, argmin returns 0.
    
    # One edge case: If an item has weight 0 and value > 0, delta might be 0?
    # If w=0, it's always taken in fractional bound. Excluding it reduces bound by v.
    # So delta = v. Score = v/v = 1.
    
    best_idx = int(np.argmin(scores))
    
    return best_idx
