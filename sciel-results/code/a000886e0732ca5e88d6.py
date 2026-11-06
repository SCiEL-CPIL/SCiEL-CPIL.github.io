"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None


def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def _greedy_fill(capacity, weights, values, density_power=1.0):
    """Fill remaining capacity using greedy by value/weight^power."""
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0, []
    
    # Compute densities
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = values / np.power(weights, density_power)
    
    # Sort by density descending
    order = np.argsort(-densities)
    
    total_value = 0.0
    total_weight = 0.0
    picked = []
    
    for idx in order:
        if weights[idx] <= capacity - total_weight + 1e-12:
            total_value += values[idx]
            total_weight += weights[idx]
            picked.append(idx)
        if total_weight >= capacity - 1e-12:
            break
    
    return total_value, picked


def _estimate_total_value(remaining_capacity, remaining_weights, remaining_values, max_lookahead=3):
    """
    Estimate the total value achievable from the current state by:
    1. Trying each feasible item as the first pick
    2. For each first pick, greedily filling the rest
    3. Also trying a few alternative greedy strategies
    """
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return 0.0, -1
    
    # Feasible items
    feasible_mask = remaining_weights <= remaining_capacity + 1e-12
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return 0.0, -1
    
    best_estimate = 0.0
    best_first_pick = -1
    
    # Try each feasible item as the first pick, then greedily fill the rest
    # To keep it fast, limit the number of candidates we evaluate
    # Use a scoring heuristic to pick top candidates
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = remaining_values[feasible_indices] / remaining_weights[feasible_indices]
    
    # Sort feasible items by density
    sorted_feasible = feasible_indices[np.argsort(-densities)]
    
    # Evaluate top K candidates as first pick
    K = min(len(sorted_feasible), max(10, n // 5))
    top_candidates = sorted_feasible[:K]
    
    # Also include some low-weight items (they might be good first picks)
    if n > 20:
        light_items = feasible_indices[np.argsort(remaining_weights[feasible_indices])[:max(5, n // 20)]]
        top_candidates = np.unique(np.concatenate([top_candidates, light_items]))
    
    for first_idx in top_candidates:
        wi = remaining_weights[first_idx]
        vi = remaining_values[first_idx]
        cap_rem = remaining_capacity - wi
        
        if cap_rem < -1e-12:
            continue
        
        # Create remaining arrays without first_idx
        mask = np.ones(n, dtype=bool)
        mask[first_idx] = False
        rem_w = remaining_weights[mask]
        rem_v = remaining_values[mask]
        
        # Greedily fill the rest
        fill_value, _ = _greedy_fill(cap_rem, rem_w, rem_v, density_power=1.0)
        
        total = vi + fill_value
        if total > best_estimate:
            best_estimate = total
            best_first_pick = int(first_idx)
    
    # Also try pure greedy without committing to a specific first item
    pure_greedy_value, _ = _greedy_fill(remaining_capacity, remaining_weights, remaining_values, density_power=1.0)
    
    # Try sqrt-density greedy
    sqrt_greedy_value, _ = _greedy_fill(remaining_capacity, remaining_weights, remaining_values, density_power=0.5)
    
    # Combine estimates
    if best_first_pick == -1:
        best_estimate = max(pure_greedy_value, sqrt_greedy_value)
        # Pick the item that greedy would pick first
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = remaining_values / remaining_weights
        best_first_pick = int(np.argmax(densities * feasible_mask))
        if not feasible_mask[best_first_pick]:
            best_first_pick = -1
    else:
        # Use the max of the lookahead estimate and pure greedy
        # But keep the first pick from lookahead if it's better
        if pure_greedy_value > best_estimate:
            # Pure greedy is better; use its first pick
            with np.errstate(divide='ignore', invalid='ignore'):
                densities = remaining_values / remaining_weights
            first_in_greedy = int(np.argmax(densities * feasible_mask))
            if feasible_mask[first_in_greedy]:
                best_first_pick = first_in_greedy
                best_estimate = pure_greedy_value
        if sqrt_greedy_value > best_estimate:
            with np.errstate(divide='ignore', invalid='ignore'):
                densities_sqrt = remaining_values / np.sqrt(remaining_weights)
            first_in_sqrt = int(np.argmax(densities_sqrt * feasible_mask))
            if feasible_mask[first_in_sqrt]:
                best_first_pick = first_in_sqrt
                best_estimate = sqrt_greedy_value
    
    return best_estimate, best_first_pick


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Check for feasible items
    feasible_mask = remaining_weights <= remaining_capacity + 1e-12
    if not np.any(feasible_mask):
        return -1
    
    # Deadline guard
    elapsed = _get_elapsed()
    if elapsed > 130.0:
        # Fallback to simple greedy
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = remaining_values / remaining_weights
        densities[~feasible_mask] = -np.inf
        return int(np.argmax(densities))
    
    # Estimate total value and pick the best first item
    estimate, best_pick = _estimate_total_value(remaining_capacity, remaining_weights, remaining_values)
    
    if best_pick == -1:
        # Fallback
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = remaining_values / remaining_weights
        densities[~feasible_mask] = -np.inf
        best_pick = int(np.argmax(densities))
        if not feasible_mask[best_pick]:
            return -1
    
    return best_pick
# EVOLVE-BLOCK-END
