"""Improved 0/1 knapsack heuristic with lookahead greedy completion."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.monotonic()
_TOTAL_BUDGET = 100.0  # safety margin under 120s


def _greedy_completion(capacity, weights, values):
    """Estimate total value by greedy density completion."""
    if capacity <= 0 or len(weights) == 0:
        return 0.0
    feasible = weights <= capacity
    if not np.any(feasible):
        return 0.0
    f_w = weights[feasible]
    f_v = values[feasible]
    densities = f_v / f_w
    order = np.argsort(-densities)
    total_val = 0.0
    rem_cap = capacity
    for idx in order:
        w = f_w[idx]
        v = f_v[idx]
        if w <= rem_cap:
            total_val += v
            rem_cap -= w
    return total_val


def _greedy_completion_value(capacity, weights, values):
    """Estimate total value by greedy value-first completion."""
    if capacity <= 0 or len(weights) == 0:
        return 0.0
    feasible = weights <= capacity
    if not np.any(feasible):
        return 0.0
    f_w = weights[feasible]
    f_v = values[feasible]
    order = np.argsort(-f_v)
    total_val = 0.0
    rem_cap = capacity
    for idx in order:
        w = f_w[idx]
        v = f_v[idx]
        if w <= rem_cap:
            total_val += v
            rem_cap -= w
    return total_val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Deadline guard
    if time.monotonic() - _START > _TOTAL_BUDGET:
        return -1

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Find feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    f_w = remaining_weights[feasible_mask]
    f_v = remaining_values[feasible_mask]

    # For each candidate, estimate completion value if we pick it now
    # Score = value of this item + estimated completion of the rest
    best_score = -1.0
    best_idx = -1

    # Use a blend: 0.7 * density_completion + 0.3 * value_completion
    for i in range(len(feasible_indices)):
        w_i = f_w[i]
        v_i = f_v[i]

        # Remaining capacity after picking item i
        rem_cap = remaining_capacity - w_i

        # Remove item i from the remaining set for completion estimation
        # Create a boolean mask of which items are NOT item i
        if len(feasible_indices) == 1:
            comp_density = 0.0
            comp_value = 0.0
        else:
            # Build arrays excluding item i
            mask = np.ones(len(f_w), dtype=bool)
            mask[i] = False
            rw = f_w[mask]
            rv = f_v[mask]
            comp_density = _greedy_completion(rem_cap, rw, rv)
            comp_value = _greedy_completion_value(rem_cap, rw, rv)

        score = v_i + 0.7 * comp_density + 0.3 * comp_value

        if score > best_score:
            best_score = score
            best_idx = feasible_indices[i]

    if best_idx == -1:
        return -1

    return int(best_idx)
# EVOLVE-BLOCK-END
