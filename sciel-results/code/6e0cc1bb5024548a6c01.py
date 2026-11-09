"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


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

    # Filter to feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    feasible_idx = np.where(feasible_mask)[0]
    fw = remaining_weights[feasible_idx]
    fv = remaining_values[feasible_idx]
    fn = len(fw)

    if fn == 1:
        return int(feasible_idx[0])

    # 1-step lookahead: for each feasible item, simulate greedy completion
    # and pick the item that yields the highest total value.
    # For large n, limit lookahead to top-k by density to save time.
    k = min(fn, 15)
    densities = fv / fw
    top_k_local = np.argsort(densities)[::-1][:k]

    best_total = -1.0
    best_local = top_k_local[0]

    for i in top_k_local:
        # Remove item i, complete greedily with rest
        mask = np.ones(fn, dtype=bool)
        mask[i] = False
        total = fv[i] + _greedy_complete(
            remaining_capacity - fw[i], fw[mask], fv[mask]
        )
        if total > best_total:
            best_total = total
            best_local = i

    return int(feasible_idx[best_local])
# EVOLVE-BLOCK-END
