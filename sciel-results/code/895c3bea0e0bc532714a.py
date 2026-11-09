"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 110.0  # safety margin before 120s kill


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


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Check time budget
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to simple greedy
        feasible = remaining_weights <= remaining_capacity
        if not np.any(feasible):
            return -1
        idxs = np.where(feasible)[0]
        densities = remaining_values[idxs] / remaining_weights[idxs]
        return int(idxs[np.argmax(densities)])
    
    # Find all feasible items
    feasible = remaining_weights <= remaining_capacity
    if not np.any(feasible):
        return -1
    feasible_idxs = np.where(feasible)[0]
    
    # If few items, just use simple greedy
    if len(feasible_idxs) <= 5:
        densities = remaining_values[feasible_idxs] / remaining_weights[feasible_idxs]
        return int(feasible_idxs[np.argmax(densities)])
    
    # 1-step lookahead: evaluate top-k candidates by their downstream greedy value
    # Sort feasible items by density to get top candidates
    densities = remaining_values[feasible_idxs] / remaining_weights[feasible_idxs]
    # Get top-k candidates (k scales with problem size, capped for speed)
    k = min(10, len(feasible_idxs))
    top_k_local = np.argsort(densities)[-k:]
    top_k_global = feasible_idxs[top_k_local]
    
    best_score = -1.0
    best_idx = -1
    
    for idx in top_k_global:
        # Simulate picking this item, then complete greedily
        new_cap = remaining_capacity - remaining_weights[idx]
        if new_cap < 0:
            continue
        
        # Create remaining arrays without this item
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        rem_w = remaining_weights[mask]
        rem_v = remaining_values[mask]
        
        # Filter to feasible only for speed
        feasible_mask = rem_w <= new_cap
        if not np.any(feasible_mask):
            score = remaining_values[idx]
        else:
            score = remaining_values[idx] + _greedy_complete(new_cap, rem_w[feasible_mask], rem_v[feasible_mask])
        
        if score > best_score:
            best_score = score
            best_idx = int(idx)
    
    if best_idx == -1:
        # Fallback
        return int(feasible_idxs[np.argmax(densities)])
    
    return best_idx
# EVOLVE-BLOCK-END
