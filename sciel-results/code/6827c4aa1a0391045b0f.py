"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard: evaluator kills after 150s. We keep a safety margin.
_START_TIME = time.time()
_DEADLINE_SECONDS = 140.0  # 10s safety margin before the 150s kill

def _elapsed_ok():
    return (time.time() - _START_TIME) < _DEADLINE_SECONDS

def _greedy_fill_fast(capacity, weights, values, order_indices, exclude_idx=-1):
    """
    Fast greedy knapsack fill using the given order of indices.
    Returns the total value achieved.
    """
    sim_cap = capacity
    sim_val = 0.0
    for idx in order_indices:
        if idx == exclude_idx:
            continue
        w = weights[idx]
        if w <= sim_cap:
            sim_cap -= w
            sim_val += values[idx]
        if sim_cap <= 0:
            break
    return sim_val

def _estimate_future_value(capacity, weights, values, exclude_idx=-1):
    """
    Estimate the future value achievable after excluding one item, using
    multiple greedy strategies. This is more realistic than the fractional
    knapsack bound for 0/1 knapsack.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    # Filter to feasible items
    feasible_mask = (weights <= capacity)
    if exclude_idx >= 0:
        feasible_mask[exclude_idx] = False
    
    if not np.any(feasible_mask):
        return 0.0
    
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return 0.0
    
    # Get weights and values for feasible items only
    fw = weights[feasible_indices]
    fv = values[feasible_indices]
    
    # Strategy 1: density order
    with np.errstate(divide='ignore', invalid='ignore'):
        dens = np.where(fw > 0, fv / fw, 0.0)
    order_density = feasible_indices[np.argsort(-dens)]
    
    # Strategy 2: value order
    order_value = feasible_indices[np.argsort(-fv)]
    
    # Strategy 3: weight order (lightest first)
    order_weight = feasible_indices[np.argsort(fw)]
    
    # Simulate all three and take the best
    val_1 = _greedy_fill_fast(capacity, weights, values, order_density, exclude_idx)
    val_2 = _greedy_fill_fast(capacity, weights, values, order_value, exclude_idx)
    val_3 = _greedy_fill_fast(capacity, weights, values, order_weight, exclude_idx)
    
    return max(val_1, val_2, val_3)

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Safety check: if we've been running too long, fall back to simple greedy
    if not _elapsed_ok():
        # Fallback: simple density greedy
        if len(remaining_weights) == 0 or remaining_capacity <= 0:
            return -1
        weights = np.asarray(remaining_weights, dtype=float)
        values = np.asarray(remaining_values, dtype=float)
        feasible = np.where(weights <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        # Pick best density among feasible
        with np.errstate(divide='ignore', invalid='ignore'):
            dens = np.where(weights[feasible] > 0, values[feasible] / weights[feasible], 0.0)
        best_local = np.argmax(dens)
        return int(feasible[best_local])

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    c = float(remaining_capacity)
    
    # Filter to feasible items
    feasible_mask = weights <= c
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
    
    # Pre-compute sorting orders for multiple greedy strategies
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(weights > 0, values / weights, 0.0)
    
    order_density = np.argsort(-densities)
    order_value = np.argsort(-values)
    order_weight = np.argsort(weights)
    
    best_total_val = -1.0
    best_idx = -1
    
    # For each feasible item, compute a realistic future value estimate
    # using multiple greedy strategies (density, value, weight)
    
    for idx in feasible_indices:
        if not _elapsed_ok():
            break
            
        w_i = weights[idx]
        v_i = values[idx]
        remaining_cap = c - w_i
        
        if remaining_cap < 0:
            continue
        
        # Estimate future value using realistic greedy simulation
        # This is more accurate than the fractional knapsack bound for 0/1 knapsack
        future_val = _estimate_future_value(remaining_cap, weights, values, idx)
        
        # Total estimated value if we pick this item now
        total_est = v_i + future_val
        
        if total_est > best_total_val:
            best_total_val = total_est
            best_idx = int(idx)
    
    if best_idx == -1:
        return -1
        
    return best_idx
# EVOLVE-BLOCK-END
