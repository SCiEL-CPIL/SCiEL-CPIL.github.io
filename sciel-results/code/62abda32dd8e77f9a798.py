"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.monotonic()
_TOTAL_BUDGET = 110.0  # safety margin before 120s limit


def _greedy_completion_value(remaining_capacity, weights, values, exclude_idx=None):
    """Estimate total value achievable by greedy density completion, optionally excluding one item."""
    if remaining_capacity <= 0 or len(weights) == 0:
        return 0.0
    
    # Create mask of feasible items, excluding the one we're evaluating
    if exclude_idx is not None:
        mask = np.ones(len(weights), dtype=bool)
        mask[exclude_idx] = False
        mask &= (weights <= remaining_capacity)
    else:
        mask = (weights <= remaining_capacity)
    
    if not np.any(mask):
        return 0.0
    
    f_w = weights[mask]
    f_v = values[mask]
    
    # Sort by density descending
    densities = f_v / f_w
    order = np.argsort(-densities)
    
    total_val = 0.0
    rem_cap = remaining_capacity
    for idx in order:
        w = f_w[idx]
        v = f_v[idx]
        if w <= rem_cap:
            total_val += v
            rem_cap -= w
        else:
            break
    return total_val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Deadline guard
    if time.monotonic() - _START > _TOTAL_BUDGET:
        return -1
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Find feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    
    # For each feasible item, estimate total value if we pick it now + greedy completion
    best_idx = -1
    best_estimated_value = -1.0
    
    for idx in feasible_indices:
        w = remaining_weights[idx]
        v = remaining_values[idx]
        
        # Value if we take this item + greedy fill the rest
        remaining_cap_after = remaining_capacity - w
        completion_val = _greedy_completion_value(
            remaining_cap_after, remaining_weights, remaining_values, exclude_idx=idx
        )
        total_est = v + completion_val
        
        if total_est > best_estimated_value:
            best_estimated_value = total_est
            best_idx = idx
    
    return int(best_idx) if best_idx >= 0 else -1
# EVOLVE-BLOCK-END
