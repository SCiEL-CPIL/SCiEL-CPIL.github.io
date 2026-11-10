"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def _fractional_knapsack_solution(capacity, weights, values):
    """
    Compute the fractional knapsack solution and identify the critical item.
    Returns the index of the item that would be fractionally included (the 'critical' item),
    or -1 if all items fit or no items.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return -1
    
    # Compute value density for each item
    with np.errstate(divide='ignore', invalid='ignore'):
        density = values / weights
    density[weights <= 0] = np.inf
    
    # Sort by density descending
    order = np.argsort(-density)
    
    remaining_cap = capacity
    critical_idx = -1
    
    for idx in order:
        w = weights[idx]
        if w <= 0:
            # Item with zero weight: always take it fully
            remaining_cap -= w
            continue
        if remaining_cap >= w:
            remaining_cap -= w
        else:
            # This is the critical item that gets fractionally filled
            critical_idx = idx
            break
    
    return critical_idx


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Check time budget
    elapsed = time.time() - _get_start_time()
    if elapsed > 140:  # Safety margin before 150s limit
        return -1
    
    # Compute the fractional knapsack critical item
    critical_idx = _fractional_knapsack_solution(remaining_capacity, remaining_weights, remaining_values)
    
    if critical_idx >= 0:
        # The critical item is the one that would be fractionally included.
        # In the 0/1 knapsack, we should prioritize this item because it's on the boundary.
        # However, we also want to consider items with high density that fit well.
        # Strategy: Return the critical item if it has reasonable density,
        # otherwise fall back to best density item that fits.
        
        # Check if the critical item is feasible (it should be by construction, but verify)
        if remaining_weights[critical_idx] <= remaining_capacity:
            return int(critical_idx)
    
    # Fallback: pick the item with highest value density that fits
    feasible = remaining_weights <= remaining_capacity
    if not np.any(feasible):
        return -1
    
    feasible_indices = np.where(feasible)[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.zeros(n)
        densities[feasible] = remaining_values[feasible] / remaining_weights[feasible]
    
    best_idx = int(np.argmax(densities[feasible_indices]))
    return int(feasible_indices[best_idx])
# EVOLVE-BLOCK-END
