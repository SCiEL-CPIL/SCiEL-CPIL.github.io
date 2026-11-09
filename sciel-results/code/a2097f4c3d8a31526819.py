"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for safety
_START_TIME = time.time()
_DEADLINE_SECONDS = 120.0
_SAFETY_MARGIN = 2.0

def _elapsed_ok():
    return (time.time() - _START_TIME) < (_DEADLINE_SECONDS - _SAFETY_MARGIN)

def _simulate_greedy(capacity, weights, values, order_indices, exclude_idx):
    """
    Simulates a greedy knapsack solution given a specific ordering of items.
    Returns the total value.
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
        dens = values[feasible] / weights[feasible]
        best_local = np.argmax(dens)
        return int(feasible[best_local])

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    
    # Filter to feasible items
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])

    # Pre-compute sorting orders for two different greedy strategies
    # 1. Density (Value / Weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(weights > 0, values / weights, 0.0)
    order_density = np.argsort(-densities)
    
    # 2. Value (Highest Value first)
    order_value = np.argsort(-values)

    best_total_val = -1.0
    best_idx = -1
    
    # For each feasible item, estimate the total value if we pick it now
    # and then fill the rest with the best greedy strategy
    for idx in feasible_indices:
        if not _elapsed_ok():
            break
            
        w_i = weights[idx]
        v_i = values[idx]
        remaining_cap = remaining_capacity - w_i
        
        # Simulate Strategy 1: Density Greedy
        val_sim_1 = _simulate_greedy(remaining_cap, weights, values, order_density, idx)
        
        # Simulate Strategy 2: Value Greedy
        val_sim_2 = _simulate_greedy(remaining_cap, weights, values, order_value, idx)
        
        # The best we can hope for with this item is the max of the two simulations
        # (In reality, the evaluator will use the specific order we pick, but since we
        # are building the solution sequentially, we assume the *next* picks will also
        # be optimized. However, since the evaluator calls this function repeatedly,
        # we just need to pick the item that leads to the best *immediate* outlook.
        # Using the max of two strong heuristics is a robust proxy for potential value.)
        
        total_val = v_i + max(val_sim_1, val_sim_2)
        
        if total_val > best_total_val:
            best_total_val = total_val
            best_idx = idx
            
    if best_idx == -1:
        return -1
        
    return int(best_idx)
# EVOLVE-BLOCK-END
