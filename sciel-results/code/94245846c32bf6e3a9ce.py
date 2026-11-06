"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s margin before 150s limit


def _fractional_knapsack_bound(weights, values, capacity):
    """
    Compute the fractional knapsack upper bound for given weights, values, capacity.
    This is the optimal value if items could be fractionally taken, sorted by density.
    """
    if capacity <= 0 or len(weights) == 0:
        return 0.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = values / weights
    densities[~np.isfinite(densities)] = 0.0
    
    # Sort by density descending
    sorted_order = np.argsort(-densities)
    sorted_weights = weights[sorted_order]
    sorted_values = values[sorted_order]
    
    total_value = 0.0
    remaining_cap = capacity
    
    for w, v in zip(sorted_weights, sorted_values):
        if w <= remaining_cap:
            total_value += v
            remaining_cap -= w
        else:
            # Take fraction of this item
            total_value += v * (remaining_cap / w)
            remaining_cap = 0.0
            break
    
    return total_value


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack using Lagrangian relaxation / fractional bound lookahead.
    For each candidate, compute candidate_value + fractional_knapsack_bound(remainder).
    Pick the candidate that maximizes this sum.
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
    
    # Consider top K candidates for lookahead
    K = min(50, len(sorted_indices))
    
    best_item_idx = -1
    best_score = -1.0
    
    for i in range(K):
        candidate_idx = sorted_indices[i]
        candidate_weight = sorted_weights[i]
        candidate_value = sorted_values[i]
        
        # Compute fractional bound for the remaining items after taking candidate i
        cap_after = remaining_capacity - candidate_weight
        
        if cap_after <= 0:
            score = candidate_value
        else:
            # Build arrays excluding item i
            # Items 0..i-1 and i+1..n-1
            mask = np.ones(len(sorted_weights), dtype=bool)
            mask[i] = False
            rem_weights = sorted_weights[mask]
            rem_values = sorted_values[mask]
            
            # Compute fractional knapsack bound
            frac_bound = _fractional_knapsack_bound(rem_weights, rem_values, cap_after)
            score = candidate_value + frac_bound
        
        if score > best_score:
            best_score = score
            best_item_idx = candidate_idx
        
        # Time check
        if time.time() > _DEADLINE - 5.0:
            break
    
    # If lookahead didn't find anything good, fall back to highest density
    if best_item_idx == -1:
        best_item_idx = sorted_indices[0]
    
    return int(best_item_idx)
# EVOLVE-BLOCK-END
