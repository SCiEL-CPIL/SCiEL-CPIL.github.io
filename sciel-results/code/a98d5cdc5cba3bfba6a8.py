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
    n = len(remaining_values)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Check time budget
    if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
        # Fall back to simple density
        feasible = remaining_weights <= remaining_capacity
        if not np.any(feasible):
            return -1
        idxs = np.where(feasible)[0]
        densities = remaining_values[idxs] / remaining_weights[idxs]
        return int(idxs[np.argmax(densities)])

    # Get feasible items
    feasible = remaining_weights <= remaining_capacity
    if not np.any(feasible):
        return -1
    feasible_idxs = np.where(feasible)[0]

    # 1-step lookahead: for each feasible candidate, simulate greedy completion
    # after picking that candidate, pick the one with highest total value
    best_score = -1.0
    best_idx = feasible_idxs[np.argmax(remaining_values[feasible_idxs] / remaining_weights[feasible_idxs])]

    for cand in feasible_idxs:
        if time.time() - _START_TIME > _TIME_LIMIT - 3.0:
            break
        # Simulate: pick cand, then greedily fill the rest
        new_cap = remaining_capacity - remaining_weights[cand]
        if new_cap <= 0:
            total = remaining_values[cand]
        else:
            # Build arrays excluding cand
            mask = np.ones(n, dtype=bool)
            mask[cand] = False
            new_w = remaining_weights[mask]
            new_v = remaining_values[mask]
            total = remaining_values[cand] + _greedy_complete(new_cap, new_w, new_v)

        if total > best_score:
            best_score = total
            best_idx = cand

    return int(best_idx)
# EVOLVE-BLOCK-END
