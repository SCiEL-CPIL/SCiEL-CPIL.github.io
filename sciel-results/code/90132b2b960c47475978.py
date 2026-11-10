"""Improved 0/1 knapsack heuristic with 1-step look-ahead (opportunity cost)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 120.0  # seconds; evaluator kills at 150s, keep safety margin
_start_time = [time.time()]  # mutable holder so we can reset per instance if needed


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Pick the next item index into the remaining (feasible) arrays, or -1 to stop.
    Uses a 1-step look-ahead: for each candidate item i, estimate total value as
    value[i] + (capacity - weight[i]) * max_density(remaining items excluding i).
    This accounts for the opportunity cost of using capacity on item i.
    """
    # Deadline guard
    if time.time() - _start_time[0] > _DEADLINE - 2.0:
        return -1

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)

    # Feasibility: only consider items that fit
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    f_weights = weights[feasible_mask]
    f_values = values[feasible_mask]
    n_f = len(f_weights)

    if n_f == 0:
        return -1

    # If only one feasible item, take it
    if n_f == 1:
        return int(feasible_indices[0])

    # Compute densities for all feasible items
    densities = f_values / f_weights

    # For each candidate item k in feasible set:
    #   score_k = f_values[k] + (remaining_capacity - f_weights[k]) * max_density_excluding_k
    # We approximate max_density_excluding_k by the global max density,
    # unless the max-density item IS k, in which case use the 2nd-highest density.

    # Sort densities to get top-2
    sorted_idx = np.argsort(densities)[::-1]
    max_density = densities[sorted_idx[0]]
    second_max_density = densities[sorted_idx[1]] if n_f > 1 else 0.0
    max_density_item = sorted_idx[0]

    scores = np.empty(n_f, dtype=np.float64)
    for k in range(n_f):
        cap_after = remaining_capacity - f_weights[k]
        if cap_after <= 0:
            # No capacity left after taking this item
            fill_density = 0.0
        else:
            if k == max_density_item:
                fill_density = second_max_density
            else:
                fill_density = max_density
        scores[k] = f_values[k] + cap_after * fill_density

    best_local = int(np.argmax(scores))
    return int(feasible_indices[best_local])
# EVOLVE-BLOCK-END
