"""0/1 Knapsack heuristic: lookahead with LP-relaxation completion and top-K candidate pruning."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.monotonic()
_TOTAL_BUDGET = 112.0  # safety margin before 120s limit


def _remaining_budget():
    return max(0.0, _TOTAL_BUDGET - (time.monotonic() - _START))


def _lp_relaxation_value(remaining_capacity, weights, values, exclude_idx=None):
    """
    Estimate total value via LP relaxation (fractional knapsack) of remaining items.
    Optionally exclude one item (the candidate being evaluated).
    """
    if remaining_capacity <= 0 or len(weights) == 0:
        return 0.0

    n = len(weights)
    if exclude_idx is not None:
        mask = np.ones(n, dtype=bool)
        mask[exclude_idx] = False
        mask &= (weights <= remaining_capacity)
    else:
        mask = (weights <= remaining_capacity)

    if not np.any(mask):
        return 0.0

    f_w = weights[mask]
    f_v = values[mask]

    if len(f_w) == 0:
        return 0.0

    # Sort by density descending
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(f_w > 0, f_v / f_w, np.inf)
    order = np.argsort(-densities)

    total_val = 0.0
    cur_cap = remaining_capacity

    for idx in order:
        w = f_w[idx]
        v = f_v[idx]
        if w <= cur_cap:
            total_val += v
            cur_cap -= w
            if cur_cap <= 0:
                break
        else:
            # Fractional part
            if w > 0:
                total_val += v * (cur_cap / w)
            break

    return total_val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select next item using lookahead: for each top-K candidate by density,
    estimate total value = candidate value + LP relaxation of remaining capacity.
    """
    # Deadline guard
    if _remaining_budget() < 0.5:
        # Fallback: simple density greedy
        w = np.asarray(remaining_weights, dtype=np.float64)
        v = np.asarray(remaining_values, dtype=np.float64)
        cap = float(remaining_capacity)
        if len(w) == 0 or cap <= 0:
            return -1
        feasible = w <= cap
        if not np.any(feasible):
            return -1
        with np.errstate(divide='ignore', invalid='ignore'):
            density = np.where(w > 0, v / w, -np.inf)
        density[~feasible] = -np.inf
        best = int(np.argmax(density))
        return best if feasible[best] else -1

    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)
    cap = float(remaining_capacity)
    n = len(w)

    if n == 0 or cap <= 0:
        return -1

    # Find feasible items
    feasible_mask = w <= cap
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]

    # Compute density for all items
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(w > 0, v / w, -np.inf)

    # Sort feasible indices by density descending
    feasible_densities = densities[feasible_indices]
    sorted_order = np.argsort(-feasible_densities)
    sorted_feasible = feasible_indices[sorted_order]

    # Adaptive K: more candidates when more time available
    budget = _remaining_budget()
    if budget > 30.0:
        K = min(len(sorted_feasible), 25)
    elif budget > 15.0:
        K = min(len(sorted_feasible), 18)
    elif budget > 8.0:
        K = min(len(sorted_feasible), 12)
    else:
        K = min(len(sorted_feasible), 8)

    best_idx = -1
    best_estimated_value = -1.0

    for i in range(K):
        idx = sorted_feasible[i]
        wi = w[idx]
        vi = v[idx]

        rem_cap = cap - wi
        if rem_cap < 0:
            continue

        # LP relaxation of remaining items, excluding this candidate
        completion_val = _lp_relaxation_value(rem_cap, w, v, exclude_idx=idx)
        total_est = vi + completion_val

        if total_est > best_estimated_value:
            best_estimated_value = total_est
            best_idx = idx

        # Time check inside loop
        if _remaining_budget() < 1.0:
            break

    return int(best_idx) if best_idx >= 0 else -1
# EVOLVE-BLOCK-END
