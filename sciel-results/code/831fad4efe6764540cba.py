import numpy as np
import time

_START = time.monotonic()
_TIME_LIMIT = 110.0  # Safety margin before the 150s hard kill


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

    sort_by_density = np.argsort(-densities)
    sort_by_value = np.argsort(-values)
    sort_by_weight = np.argsort(weights)

    # For fast membership test
    feasible_set = set(feasible_indices.tolist())

    # Greedy fill helper: given remaining capacity, a sorted order of all indices,
    # and an excluded index (the candidate we already "picked"), fill greedily.
    def greedy_fill(rem_cap, sorted_indices, excluded_idx):
        total_val = 0.0
        current_cap = rem_cap
        for idx in sorted_indices:
            if idx == excluded_idx:
                continue
            if idx not in feasible_set:
                continue
            w = weights[idx]
            if w <= current_cap + 1e-12:
                total_val += values[idx]
                current_cap -= w
                if current_cap <= 1e-9:
                    break
        return total_val

    best_value = -1.0
    best_idx = -1

    # Evaluate each feasible candidate as the first pick, then complete greedily
    # under three different completion heuristics; take the best estimate.
    for i in feasible_indices:
        rem_cap_after_i = remaining_capacity - weights[i]

        if rem_cap_after_i < -1e-12:
            continue

        if rem_cap_after_i < 0:
            rem_cap_after_i = 0.0

        # Three completion policies
        val1 = values[i] + greedy_fill(rem_cap_after_i, sort_by_density, i)
        val2 = values[i] + greedy_fill(rem_cap_after_i, sort_by_value, i)
        val3 = values[i] + greedy_fill(rem_cap_after_i, sort_by_weight, i)

        best_estimate = max(val1, val2, val3)

        if best_estimate > best_value:
            best_value = best_estimate
            best_idx = i

    if best_idx == -1:
        return -1

    return int(best_idx)
