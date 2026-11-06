import numpy as np
import time

_START = time.monotonic()
_TIME_LIMIT = 110.0  # Safety margin before the 150s hard kill


def _vectorized_greedy_fill(rem_cap, sorted_indices, weights, values, excluded_idx, feasible_mask):
    """
    Vectorized greedy fill: walk through sorted_indices in order,
    picking items that fit. Returns total value.
    """
    if rem_cap <= 1e-9:
        return 0.0

    # Pre-filter: only consider feasible, non-excluded indices in sorted order
    mask = feasible_mask.copy()
    mask[excluded_idx] = False
    candidates = sorted_indices[mask[sorted_indices]]

    if len(candidates) == 0:
        return 0.0

    # Walk through candidates, accumulating until capacity is exhausted
    total_val = 0.0
    current_cap = rem_cap

    # Use numpy for the main loop but we still need sequential logic.
    # However, we can at least avoid the Python set lookup by using the mask.
    for idx in candidates:
        w = weights[idx]
        if w <= current_cap + 1e-12:
            total_val += values[idx]
            current_cap -= w
            if current_cap <= 1e-9:
                break

    return total_val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Deadline guard
    if time.monotonic() - _START > _TIME_LIMIT:
        return -1

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)

    # Feasible items only
    feasible_mask = weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]

    if len(feasible_indices) == 0:
        return -1

    # Sorting orders
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = values / weights
    densities[weights == 0] = np.inf

    # Additional heuristic: value / weight^2 (favors lighter high-value items)
    with np.errstate(divide='ignore', invalid='ignore'):
        density_sq = values / (weights * weights)
    density_sq[weights == 0] = np.inf

    sort_by_density = np.argsort(-densities)
    sort_by_value = np.argsort(-values)
    sort_by_weight = np.argsort(weights)
    sort_by_density_sq = np.argsort(-density_sq)

    # Evaluate each feasible candidate as the first pick, then complete greedily
    # under four different completion heuristics; take the best estimate.
    best_value = -1.0
    best_idx = -1

    for i in feasible_indices:
        rem_cap_after_i = remaining_capacity - weights[i]

        if rem_cap_after_i < -1e-12:
            continue

        if rem_cap_after_i < 0:
            rem_cap_after_i = 0.0

        # Four completion policies
        val1 = values[i] + _vectorized_greedy_fill(rem_cap_after_i, sort_by_density, weights, values, i, feasible_mask)
        val2 = values[i] + _vectorized_greedy_fill(rem_cap_after_i, sort_by_value, weights, values, i, feasible_mask)
        val3 = values[i] + _vectorized_greedy_fill(rem_cap_after_i, sort_by_weight, weights, values, i, feasible_mask)
        val4 = values[i] + _vectorized_greedy_fill(rem_cap_after_i, sort_by_density_sq, weights, values, i, feasible_mask)

        best_estimate = max(val1, val2, val3, val4)

        if best_estimate > best_value:
            best_value = best_estimate
            best_idx = i

    if best_idx == -1:
        return -1

    return int(best_idx)
