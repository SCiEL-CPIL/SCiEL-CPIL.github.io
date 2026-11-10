"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 110.0  # safety margin before 120s kill


def _fractional_upper_bound(capacity, weights, values):
    """Compute fractional knapsack upper bound for remaining items."""
    if capacity <= 0 or len(weights) == 0:
        return 0.0
    # Only consider feasible items
    feasible = weights <= capacity
    if not np.any(feasible):
        return 0.0
    
    w = weights[feasible]
    v = values[feasible]
    
    if len(w) == 0:
        return 0.0
    
    # Sort by value density
    densities = v / w
    order = np.argsort(-densities)
    
    total_value = 0.0
    remaining_cap = capacity
    
    for idx in order:
        if remaining_cap <= 0:
            break
        if w[idx] <= remaining_cap:
            total_value += v[idx]
            remaining_cap -= w[idx]
        else:
            total_value += v[idx] * (remaining_cap / w[idx])
            break
    
    return total_value


def _greedy_complete(capacity, weights, values):
    """Complete greedily by value density; return total value."""
    if len(values) == 0 or capacity <= 0:
        return 0.0
    total = 0.0
    cap = capacity
    w = weights.copy()
    v = values.copy()
    n = len(w)
    
    for _ in range(n):
        if cap <= 0:
            break
        feasible = w <= cap
        if not np.any(feasible):
            break
        idxs = np.where(feasible)[0]
        if len(idxs) == 0:
            break
        densities = v[idxs] / w[idxs]
        best = idxs[np.argmax(densities)]
        total += v[best]
        cap -= w[best]
        w = np.delete(w, best)
        v = np.delete(v, best)
    
    return total


def _branch_and_bound_select(capacity, weights, values):
    """
    Use branch-and-bound with fractional relaxation to guide selection.
    Returns the index of the best item to select next, or -1 to stop.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return -1
    
    # Check if any item is feasible
    feasible = weights <= capacity
    if not np.any(feasible):
        return -1
    
    # For each feasible item, estimate the total value if we pick it
    # and then complete greedily by density
    feasible_idxs = np.where(feasible)[0]
    
    best_score = -1
    best_idx = -1
    
    for idx in feasible_idxs:
        # If we pick this item
        new_cap = capacity - weights[idx]
        new_val = values[idx]
        
        # Complete greedily with remaining items
        remaining_w = np.delete(weights, idx)
        remaining_v = np.delete(values, idx)
        
        greedy_val = _greedy_complete(new_cap, remaining_w, remaining_v)
        total_val = new_val + greedy_val
        
        if total_val > best_score:
            best_score = total_val
            best_idx = idx
    
    return int(best_idx)


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select the next item to add to the knapsack using branch-and-bound guided selection.
    Returns index into remaining arrays, or -1 to stop.
    """
    # Check time budget
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return -1
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Check if any item is feasible
    feasible = remaining_weights <= remaining_capacity
    if not np.any(feasible):
        return -1
    
    # For small n, use exhaustive branch-and-bound guided selection
    # For larger n, fall back to greedy density with occasional lookahead
    if n <= 15:
        return _branch_and_bound_select(remaining_capacity, remaining_weights, remaining_values)
    else:
        # For larger instances, use a hybrid: pick from top-k by density,
        # evaluate each with greedy completion
        feasible_idxs = np.where(feasible)[0]
        densities = remaining_values[feasible_idxs] / remaining_weights[feasible_idxs]
        
        # Take top min(8, n) candidates by density
        k = min(8, len(feasible_idxs))
        top_k_local = np.argsort(-densities)[:k]
        top_k_global = feasible_idxs[top_k_local]
        
        best_score = -1
        best_idx = -1
        
        for idx in top_k_global:
            new_cap = remaining_capacity - remaining_weights[idx]
            new_val = remaining_values[idx]
            
            remaining_w = np.delete(remaining_weights, idx)
            remaining_v = np.delete(remaining_values, idx)
            
            greedy_val = _greedy_complete(new_cap, remaining_w, remaining_v)
            total_val = new_val + greedy_val
            
            if total_val > best_score:
                best_score = total_val
                best_idx = idx
        
        return int(best_idx)
# EVOLVE-BLOCK-END
