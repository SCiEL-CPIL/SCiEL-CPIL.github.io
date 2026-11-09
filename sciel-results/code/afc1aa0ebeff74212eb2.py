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


def _local_search_swap(capacity, weights, values, selected_mask, max_iters=500):
    """Try to improve solution by swapping selected items with unselected ones.
    Returns improved total value and updated selected mask."""
    total = float(np.sum(values[selected_mask]))
    cap_used = float(np.sum(weights[selected_mask]))
    remaining_cap = capacity - cap_used
    
    selected_idx = np.where(selected_mask)[0]
    unselected_idx = np.where(~selected_mask)[0]
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # Try swapping each selected item with each unselected item
        for si in selected_idx:
            if improved:
                break
            w_out = weights[si]
            v_out = values[si]
            # Find best unselected item that can replace si
            # New capacity after removing si: remaining_cap + w_out
            new_cap = remaining_cap + w_out
            # Find unselected items that fit in new_cap
            candidates = unselected_idx[weights[unselected_idx] <= new_cap]
            if len(candidates) == 0:
                continue
            # Choose the one that gives maximum value increase
            gains = values[candidates] - v_out
            if np.max(gains) <= 0:
                continue
            best_candidate = candidates[np.argmax(gains)]
            # Perform swap
            selected_mask[si] = False
            selected_mask[best_candidate] = True
            total += gains[np.argmax(gains)]
            remaining_cap = new_cap - weights[best_candidate]
            # Update index sets
            selected_idx = np.where(selected_mask)[0]
            unselected_idx = np.where(~selected_mask)[0]
            improved = True
    return total, selected_mask


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """Pick the next item to add, or -1 to stop."""
    if remaining_capacity <= 0 or len(remaining_weights) == 0:
        return -1
    
    n = len(remaining_weights)
    
    # Check time budget
    if time.time() - _START_TIME > _TIME_LIMIT - 5:
        return -1
    
    # Feasible items only
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_idx = np.where(feasible_mask)[0]
    f_w = remaining_weights[feasible_mask]
    f_v = remaining_values[feasible_mask]
    
    # Compute value density for feasible items
    densities = f_v / f_w
    
    # Base greedy: pick highest density
    best_idx_local = np.argmax(densities)
    best_idx_global = feasible_idx[best_idx_local]
    
    # Try to improve with a short lookahead/local search
    # Build initial solution by greedy
    selected = np.zeros(n, dtype=bool)
    cap = remaining_capacity
    w = remaining_weights.copy()
    v = remaining_values.copy()
    
    # Greedy fill
    for _ in range(min(n, 30)):  # Limit iterations for speed
        if cap <= 0:
            break
        feas = w <= cap
        if not np.any(feas):
            break
        idxs = np.where(feas)[0]
        dens = v[idxs] / w[idxs]
        bi = idxs[np.argmax(dens)]
        selected[bi] = True
        cap -= w[bi]
        w[bi] = np.inf  # Mark as taken
    
    # Local search to improve
    if time.time() - _START_TIME < _TIME_LIMIT - 10:
        total, selected = _local_search_swap(remaining_capacity, remaining_weights, remaining_values, selected, max_iters=100)
    
    # The first item in the improved solution is our pick
    # But we need to return an index into the remaining arrays
    # Find the first selected item in order of density preference
    if np.any(selected):
        # Prefer the highest density item among selected
        sel_idx = np.where(selected)[0]
        if len(sel_idx) > 0:
            # Pick the one with highest density among selected
            sel_densities = remaining_values[sel_idx] / remaining_weights[sel_idx]
            best_sel = sel_idx[np.argmax(sel_densities)]
            return int(best_sel)
    
    # Fallback to base greedy
    return int(best_idx_global)
# EVOLVE-BLOCK-END
