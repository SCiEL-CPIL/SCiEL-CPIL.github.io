"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = None
_TIME_LIMIT = 115.0  # safety margin before 120s limit


def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _check_time():
    """Return True if we should continue (time remaining > 1s margin)."""
    if _START_TIME is None:
        return True
    return (time.time() - _START_TIME) < _TIME_LIMIT


def _greedy_fill(capacity, weights, values):
    """
    Greedy fill: sort by density descending, pick items that fit.
    Returns total value.
    """
    if capacity <= 0 or len(weights) == 0:
        return 0.0
    
    valid = weights <= capacity
    if not np.any(valid):
        return 0.0
    
    w = weights[valid]
    v = values[valid]
    
    if len(w) == 0:
        return 0.0
    
    density = v / w
    order = np.argsort(-density)
    
    total_value = 0.0
    remaining_cap = capacity
    
    for i in order:
        wi = w[i]
        if wi <= remaining_cap:
            remaining_cap -= wi
            total_value += v[i]
            if remaining_cap <= 0:
                break
    
    return total_value


def _two_step_lookahead(remaining_capacity, remaining_weights, remaining_values, first_idx):
    """
    For a given first pick, find the best second pick, then greedy fill the rest.
    Returns the total value for the best two-step choice.
    """
    n = len(remaining_weights)
    cap_after_first = remaining_capacity - remaining_weights[first_idx]
    
    if cap_after_first <= 0:
        return remaining_values[first_idx]
    
    # Find all items that could be the second pick (excluding first_idx)
    mask = np.ones(n, dtype=bool)
    mask[first_idx] = False
    candidate_indices = np.where(mask)[0]
    
    if len(candidate_indices) == 0:
        return remaining_values[first_idx]
    
    # Filter to items that fit after first pick
    fits_second = remaining_weights[candidate_indices] <= cap_after_first
    second_candidates = candidate_indices[fits_second]
    
    if len(second_candidates) == 0:
        # No second item fits, just greedy fill remaining capacity with all items except first
        mask2 = np.ones(n, dtype=bool)
        mask2[first_idx] = False
        idxs = np.where(mask2)[0]
        if len(idxs) == 0:
            return remaining_values[first_idx]
        w_sub = remaining_weights[idxs]
        v_sub = remaining_values[idxs]
        return remaining_values[first_idx] + _greedy_fill(cap_after_first, w_sub, v_sub)
    
    # For each possible second pick, simulate greedy fill of the rest
    # Limit candidates for speed
    if len(second_candidates) > 30:
        # Sort by density and take top 30
        dens = remaining_values[second_candidates] / remaining_weights[second_candidates]
        top_order = np.argsort(-dens)
        second_candidates = second_candidates[top_order[:30]]
    
    best_total = -1.0
    
    for second_idx in second_candidates:
        cap_after_second = cap_after_first - remaining_weights[second_idx]
        
        if cap_after_second <= 0:
            total_val = remaining_values[first_idx] + remaining_values[second_idx]
        else:
            # Greedy fill with all items except first and second
            mask2 = np.ones(n, dtype=bool)
            mask2[first_idx] = False
            mask2[second_idx] = False
            idxs = np.where(mask2)[0]
            
            if len(idxs) == 0:
                total_val = remaining_values[first_idx] + remaining_values[second_idx]
            else:
                w_sub = remaining_weights[idxs]
                v_sub = remaining_values[idxs]
                greedy_val = _greedy_fill(cap_after_second, w_sub, v_sub)
                total_val = remaining_values[first_idx] + remaining_values[second_idx] + greedy_val
        
        if total_val > best_total:
            best_total = total_val
    
    return best_total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Pick the next item to add to the knapsack, or -1 to stop.
    
    Strategy: 
    - 1-step lookahead for all feasible candidates: pick item, then greedy fill rest.
    - 2-step lookahead for top candidates: pick item i, then best item j, then greedy fill.
    Pick the item that leads to the best total value.
    """
    _init_time()
    
    if remaining_capacity <= 0:
        return -1
    
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # If only one item, pick it if it fits
    if n == 1:
        if remaining_weights[0] <= remaining_capacity:
            return 0
        return -1
    
    # Check if any item fits
    fits = remaining_weights <= remaining_capacity
    if not np.any(fits):
        return -1
    
    # Get indices of items that fit
    fitting_idx = np.where(fits)[0]
    
    if len(fitting_idx) == 0:
        return -1
    
    # If only one item fits, pick it
    if len(fitting_idx) == 1:
        return int(fitting_idx[0])
    
    # Compute densities for all items
    densities = remaining_values / remaining_weights
    
    # Sort fitting items by density descending
    fitting_densities = densities[fitting_idx]
    sorted_order = np.argsort(-fitting_densities)
    
    # Determine candidate set
    if n <= 40:
        candidates = fitting_idx[sorted_order]
    else:
        top_k = min(25, len(fitting_idx))
        candidates = fitting_idx[sorted_order[:top_k]]
    
    # Phase 1: 1-step lookahead for all candidates
    best_value = -1.0
    best_idx = -1
    candidate_scores = {}
    
    for idx in candidates:
        if not _check_time():
            break
        
        new_cap = remaining_capacity - remaining_weights[idx]
        
        if new_cap < 0:
            continue
        
        # Greedy fill remaining items excluding idx
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        idxs = np.where(mask)[0]
        
        if len(idxs) == 0:
            total_val = remaining_values[idx]
        else:
            w_sub = remaining_weights[idxs]
            v_sub = remaining_values[idxs]
            greedy_val = _greedy_fill(new_cap, w_sub, v_sub)
            total_val = remaining_values[idx] + greedy_val
        
        candidate_scores[idx] = total_val
        if total_val > best_value:
            best_value = total_val
            best_idx = int(idx)
    
    if best_idx == -1:
        # Fallback: pick highest density item that fits
        return int(np.argmax(densities[fitting_idx]))
    
    # Phase 2: 2-step lookahead for top candidates (by 1-step score)
    # Only do this if we have time and n is manageable
    if _check_time() and n <= 60:
        # Take top 8 candidates from phase 1 by score
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = [idx for idx, score in sorted_candidates[:8]]
        
        for idx in top_candidates:
            if not _check_time():
                break
            
            # 2-step lookahead: try to find a better total by considering the second pick explicitly
            two_step_val = _two_step_lookahead(remaining_capacity, remaining_weights, remaining_values, idx)
            
            if two_step_val > best_value:
                best_value = two_step_val
                best_idx = int(idx)
    
    return best_idx
# EVOLVE-BLOCK-END
