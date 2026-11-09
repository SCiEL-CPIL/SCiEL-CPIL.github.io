"""Knapsack heuristic: 1-step lookahead over top-K density candidates."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 120.0
_START_TIME = time.time()
_K = 20


def _greedy_fill(remaining_capacity, weights, values):
    """Greedily fill by value density; return total value (and items count)."""
    if remaining_capacity <= 0 or len(weights) == 0:
        return 0.0
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    # Only feasible items
    mask = w <= remaining_capacity
    if not np.any(mask):
        return 0.0
    w = w[mask]
    v = v[mask]
    with np.errstate(divide='ignore', invalid='ignore'):
        dens = v / w
    # Sort by density descending
    order = np.argsort(-dens)
    w = w[order]
    v = v[order]
    total = 0.0
    cap = remaining_capacity
    for wi, vi in zip(w, v):
        if wi <= cap:
            cap -= wi
            total += vi
        if cap <= 0:
            break
    return total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE - 1.0:
        # Safety margin: return best feasible by density
        w = np.asarray(remaining_weights, dtype=float)
        v = np.asarray(remaining_values, dtype=float)
        mask = w <= remaining_capacity
        if not np.any(mask):
            return -1
        idx = np.where(mask)[0]
        with np.errstate(divide='ignore', invalid='ignore'):
            dens = v[idx] / w[idx]
        return int(idx[np.argmax(dens)])

    n = len(remaining_weights)
    if n == 0:
        return -1

    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    c = float(remaining_capacity)

    # Feasible items only
    mask = w <= c
    if not np.any(mask):
        return -1
    feasible_idx = np.where(mask)[0]
    wf = w[feasible_idx]
    vf = v[feasible_idx]

    # Top-K by density
    with np.errstate(divide='ignore', invalid='ignore'):
        dens = vf / wf
    # Handle inf densities (zero weight): they're great, put them first
    k = min(_K, len(feasible_idx))
    # Sort by density descending, take top K
    local_order = np.argsort(-dens)
    top_local = local_order[:k]

    best_score = -1.0
    best_local = -1

    for li in top_local:
        wi = wf[li]
        vi = vf[li]
        # Commit to this item, then greedy fill the rest
        new_cap = c - wi
        # Remaining items: all feasible except this one
        rem_mask = np.ones(len(feasible_idx), dtype=bool)
        rem_mask[li] = False
        rem_w = wf[rem_mask]
        rem_v = vf[rem_mask]
        rem_val = _greedy_fill(new_cap, rem_w, rem_v)
        score = vi + rem_val
        if score > best_score:
            best_score = score
            best_local = li

    if best_local < 0:
        return -1

    # Map local index back to original index
    return int(feasible_idx[best_local])
# EVOLVE-BLOCK-END
