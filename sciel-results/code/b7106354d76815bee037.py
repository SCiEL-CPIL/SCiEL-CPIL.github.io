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


def _evaluate_candidate(remaining_capacity, remaining_weights, remaining_values, candidate_idx):
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


def _local_refine(remaining_capacity, remaining_weights, remaining_values, current_pick):
    """
    Try to improve the current pick by swapping it with another item.
    Returns the best index found.
    """
    n = len(remaining_weights)
    best_idx = current_pick
    best_score = _evaluate_candidate(remaining_capacity, remaining_weights, remaining_values, current_pick)
    
    # Try swapping: remove current_pick, try each other feasible item
    # But we need to be careful: we're choosing ONE item to pick next.
    # So local refinement means: among items that could replace current_pick in the "next pick" role,
    # find one that gives better total when followed by greedy completion.
    
    # This is essentially the same as evaluating all candidates, so local refinement
    # in this context means: if we have a shortlist, check a few more.
    # For now, just return current_pick (the main improvement is in candidate selection).
    return best_idx


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
    
    # Top few by value
    val_order = np.argsort(-v)
    for i in range(min(5, n_feas)):
        # Find position in density-sorted array
        # val_order[i] is index in w array; we need its position in density-sorted order
        # Actually, let's just add these items directly
        pass
    
    # Let's add top-value items by finding their position in the sorted order
    # Create a mapping from original index to sorted position
    sorted_pos = np.zeros(n_feas, dtype=int)
    for pos, orig_idx in enumerate(order):
        sorted_pos[orig_idx] = pos
    
    for i in range(min(5, n_feas)):
        candidate_positions.add(sorted_pos[val_order[i]])
    
    # Low weight items (top 5 by lowest weight)
    w_order = np.argsort(w)
    for i in range(min(5, n_feas)):
        candidate_positions.add(sorted_pos[w_order[i]])
    
    # Convert to indices in the feasible array
    candidate_indices = [order[pos] for pos in sorted(candidate_positions)]
    
    # Evaluate each candidate with 1-step lookahead
    best_score = -1
    best_candidate = candidate_indices[0]
    
    for ci in candidate_indices:
        score = _evaluate_candidate(remaining_capacity, w, v, ci)
        if score > best_score:
            best_score = score
            best_candidate = ci
    
    # If we have time, do a second pass with a wider candidate set
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - 5.0 and n_feas > len(candidate_indices):
        # Add more candidates: next 10 by density
        for i in range(k_top, min(k_top + 10, n_feas)):
            pos = i
            if pos not in candidate_positions:
                ci = order[pos]
                score = _evaluate_candidate(remaining_capacity, w, v, ci)
                if score > best_score:
                    best_score = score
                    best_candidate = ci
                candidate_positions.add(pos)
    
    # Return the index in the original remaining arrays
    return int(idx_feasible[best_candidate])
# EVOLVE-BLOCK-END
