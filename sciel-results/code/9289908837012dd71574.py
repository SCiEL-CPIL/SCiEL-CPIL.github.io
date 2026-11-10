"""Improved 0/1 knapsack heuristic with multi-step lookahead and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track global start time for deadline enforcement
_START_TIME = time.time()
_DEADLINE = 115.0  # seconds; safety margin before 120s kill


def _greedy_fill_value(capacity, weights, values):
    """Greedily fill capacity by density, return total value. Assumes weights/values are numpy arrays."""
    if len(weights) == 0 or capacity <= 0:
        return 0.0
    # Filter to feasible items
    mask = weights <= capacity
    if not np.any(mask):
        return 0.0
    w = weights[mask]
    v = values[mask]
    density = v / w
    order = np.argsort(-density)
    w = w[order]
    v = v[order]
    cap = capacity
    total = 0.0
    for wi, vi in zip(w, v):
        if wi <= cap:
            total += vi
            cap -= wi
        else:
            break
    return total


def _fractional_knapsack_value(capacity, weights, values):
    """Compute the fractional knapsack optimal value (smooth surrogate for 0/1)."""
    if len(weights) == 0 or capacity <= 0:
        return 0.0
    mask = weights <= capacity
    if not np.any(mask):
        return 0.0
    w = weights[mask]
    v = values[mask]
    density = v / w
    order = np.argsort(-density)
    w = w[order]
    v = v[order]
    cap = capacity
    total = 0.0
    for wi, vi in zip(w, v):
        if wi <= cap:
            total += vi
            cap -= wi
        else:
            total += vi * (cap / wi)
            break
    return total


def _evaluate_candidate_1step(remaining_capacity, remaining_weights, remaining_values, candidate_idx):
    """
    Evaluate picking candidate_idx: simulate adding it, then greedily complete.
    Returns estimated total value.
    """
    new_cap = remaining_capacity - remaining_weights[candidate_idx]
    if new_cap < -1e-9:
        return -1.0
    
    # Build mask to remove candidate
    n = len(remaining_weights)
    mask_remove = np.ones(n, dtype=bool)
    mask_remove[candidate_idx] = False
    w_rem = remaining_weights[mask_remove]
    v_rem = remaining_values[mask_remove]
    
    # Filter feasible
    mask_feas = w_rem <= new_cap
    if not np.any(mask_feas):
        return float(remaining_values[candidate_idx])
    
    w_rem = w_rem[mask_feas]
    v_rem = v_rem[mask_feas]
    
    total = float(remaining_values[candidate_idx])
    total += _greedy_fill_value(new_cap, w_rem, v_rem)
    return total


def _evaluate_candidate_2step(remaining_capacity, remaining_weights, remaining_values, candidate_idx):
    """
    Evaluate picking candidate_idx with 2-step lookahead:
    After adding candidate_idx, try the top few next items and greedily complete.
    Returns the best estimated total value.
    """
    new_cap = remaining_capacity - remaining_weights[candidate_idx]
    if new_cap < -1e-9:
        return -1.0
    
    # Build mask to remove candidate
    n = len(remaining_weights)
    mask_remove = np.ones(n, dtype=bool)
    mask_remove[candidate_idx] = False
    w_rem = remaining_weights[mask_remove]
    v_rem = remaining_values[mask_remove]
    
    # Filter feasible
    mask_feas = w_rem <= new_cap
    if not np.any(mask_feas):
        return float(remaining_values[candidate_idx])
    
    w_rem = w_rem[mask_feas]
    v_rem = v_rem[mask_feas]
    
    base_value = float(remaining_values[candidate_idx])
    
    # Get top 5 by density from remaining
    if len(w_rem) <= 5:
        return base_value + _greedy_fill_value(new_cap, w_rem, v_rem)
    
    density = v_rem / w_rem
    top5_order = np.argsort(-density)[:5]
    
    best_total = base_value + _greedy_fill_value(new_cap, w_rem, v_rem)
    
    for second_idx in top5_order:
        second_cap = new_cap - w_rem[second_idx]
        if second_cap < -1e-9:
            continue
        
        # Remove second item
        mask2 = np.ones(len(w_rem), dtype=bool)
        mask2[second_idx] = False
        w2 = w_rem[mask2]
        v2 = v_rem[mask2]
        
        mask2_feas = w2 <= second_cap
        if not np.any(mask2_feas):
            total = base_value + float(v_rem[second_idx])
        else:
            w2 = w2[mask2_feas]
            v2 = v2[mask2_feas]
            total = base_value + float(v_rem[second_idx]) + _greedy_fill_value(second_cap, w2, v2)
        
        if total > best_total:
            best_total = total
    
    return best_total


def _score_candidate(remaining_capacity, remaining_weights, remaining_values, candidate_idx):
    """
    Combined scoring: blend 1-step greedy completion, 2-step lookahead,
    and fractional knapsack surrogate.
    """
    new_cap = remaining_capacity - remaining_weights[candidate_idx]
    if new_cap < -1e-9:
        return -1.0
    
    n = len(remaining_weights)
    mask_remove = np.ones(n, dtype=bool)
    mask_remove[candidate_idx] = False
    w_rem = remaining_weights[mask_remove]
    v_rem = remaining_values[mask_remove]
    
    mask_feas = w_rem <= new_cap
    if not np.any(mask_feas):
        return float(remaining_values[candidate_idx])
    
    w_rem = w_rem[mask_feas]
    v_rem = v_rem[mask_feas]
    
    # 1-step greedy
    greedy_val = float(remaining_values[candidate_idx]) + _greedy_fill_value(new_cap, w_rem, v_rem)
    
    # Fractional surrogate (smooth upper bound)
    frac_val = float(remaining_values[candidate_idx]) + _fractional_knapsack_value(new_cap, w_rem, v_rem)
    
    # Blend: 70% greedy, 30% fractional (fractional is an upper bound, so weight less)
    score = 0.7 * greedy_val + 0.3 * frac_val
    
    return score


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Pick the next item to add to the knapsack, or -1 to stop.
    """
    # Deadline check
    if time.time() - _START_TIME > _DEADLINE:
        return -1
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Filter to feasible items (weight <= remaining capacity)
    mask = remaining_weights <= remaining_capacity
    if not np.any(mask):
        return -1
    
    idx_feasible = np.where(mask)[0]
    w = remaining_weights[idx_feasible]
    v = remaining_values[idx_feasible]
    n_feas = len(w)
    
    if n_feas == 0:
        return -1
    
    # Compute density
    density = v / w
    
    # Sort by density descending
    order = np.argsort(-density)
    w_sorted = w[order]
    v_sorted = v[order]
    
    # Determine the fractional knapsack solution to find the "break" point
    cap = remaining_capacity
    break_idx = -1
    for i in range(len(w_sorted)):
        if w_sorted[i] <= cap:
            cap -= w_sorted[i]
        else:
            break_idx = i
            break
    
    # Build candidate set:
    # 1. Top-k by density
    # 2. Break item and its neighbors
    # 3. Items with high value (top few by value)
    # 4. Items with low weight (to leave room)
    
    k_top = 10
    candidate_positions = set()
    
    # Top k by density
    for i in range(min(k_top, n_feas)):
        candidate_positions.add(i)
    
    # Break item and neighbors
    if break_idx >= 0:
        for offset in range(-3, 4):
            j = break_idx + offset
            if 0 <= j < n_feas:
                candidate_positions.add(j)
    
    # Create a mapping from original index to sorted position
    sorted_pos = np.zeros(n_feas, dtype=int)
    for pos, orig_idx in enumerate(order):
        sorted_pos[orig_idx] = pos
    
    # Top few by value
    val_order = np.argsort(-v)
    for i in range(min(5, n_feas)):
        candidate_positions.add(sorted_pos[val_order[i]])
    
    # Low weight items (top 5 by lowest weight)
    w_order = np.argsort(w)
    for i in range(min(5, n_feas)):
        candidate_positions.add(sorted_pos[w_order[i]])
    
    # Convert to indices in the feasible array
    candidate_indices = [order[pos] for pos in sorted(candidate_positions)]
    
    # First pass: score all candidates with blended 1-step + fractional surrogate
    best_score = -1
    best_candidate = candidate_indices[0]
    scores = {}
    
    for ci in candidate_indices:
        score = _score_candidate(remaining_capacity, w, v, ci)
        scores[ci] = score
        if score > best_score:
            best_score = score
            best_candidate = ci
    
    # Second pass: for top 5 candidates by first-pass score, do 2-step lookahead
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - 5.0 and len(candidate_indices) >= 2:
        top5 = sorted(candidate_indices, key=lambda x: scores[x], reverse=True)[:5]
        for ci in top5:
            if ci == best_candidate:
                continue
            score2 = _evaluate_candidate_2step(remaining_capacity, w, v, ci)
            if score2 > best_score:
                best_score = score2
                best_candidate = ci
    
    # If we still have time, expand candidate set
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - 5.0 and n_feas > len(candidate_indices):
        for i in range(k_top, min(k_top + 10, n_feas)):
            pos = i
            if pos not in candidate_positions:
                ci = order[pos]
                score = _score_candidate(remaining_capacity, w, v, ci)
                if score > best_score:
                    best_score = score
                    best_candidate = ci
                candidate_positions.add(pos)
    
    # Return the index in the original remaining arrays
    return int(idx_feasible[best_candidate])
# EVOLVE-BLOCK-END
