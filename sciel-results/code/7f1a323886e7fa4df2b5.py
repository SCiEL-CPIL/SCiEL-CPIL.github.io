"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s margin before 150s limit


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: pick next item or -1 to stop.
    Uses a lookahead heuristic that considers future capacity impact.
    """
    if time.time() > _DEADLINE:
        return -1
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Filter to feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = remaining_weights[feasible_mask]
    feasible_values = remaining_values[feasible_mask]
    
    if len(feasible_indices) == 0:
        return -1
    
    # Calculate value density for feasible items
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = feasible_values / feasible_weights
    densities[~np.isfinite(densities)] = 0.0
    
    # Sort by density descending
    sorted_order = np.argsort(-densities)
    sorted_indices = feasible_indices[sorted_order]
    sorted_weights = feasible_weights[sorted_order]
    sorted_values = feasible_values[sorted_order]
    sorted_densities = densities[sorted_order]
    
    # Lookahead: simulate greedy packing for top candidates
    # Pick the item that leads to the best total value in a quick simulation
    best_item_idx = -1
    best_total_value = -1.0
    
    # Consider top K candidates for lookahead (K is small for speed)
    K = min(10, len(sorted_indices))
    
    for i in range(K):
        candidate_idx = sorted_indices[i]
        candidate_weight = sorted_weights[i]
        candidate_value = sorted_values[i]
        
        # Simulate: take this item first, then greedily add remaining by density
        cap_after = remaining_capacity - candidate_weight
        total_val = candidate_value
        
        # Add remaining items greedily by density (excluding the candidate)
        # We need to skip the candidate item in the sorted list
        for j in range(len(sorted_indices)):
            if j == i:
                continue
            item_w = sorted_weights[j]
            if item_w <= cap_after:
                cap_after -= item_w
                total_val += sorted_values[j]
            else:
                # Since sorted by density, if this doesn't fit, smaller ones might
                # But for speed, we can break early if weight is too large
                # Actually, we should continue because a later item might fit
                pass
        
        if total_val > best_total_value:
            best_total_value = total_val
            best_item_idx = candidate_idx
    
    # If lookahead didn't find anything good, fall back to highest density
    if best_item_idx == -1:
        best_item_idx = sorted_indices[0]
    
    return int(best_item_idx)
# EVOLVE-BLOCK-END
