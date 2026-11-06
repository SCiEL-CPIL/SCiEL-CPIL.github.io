"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 110.0  # Safety margin before 120s limit


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Pick the next item using a 1-step lookahead heuristic.
    Returns index into remaining arrays, or -1 to stop.
    """
    n = len(remaining_weights)
    if n == 0:
        return -1

    # Check time
    if time.time() - _start_time > _TIME_LIMIT:
        return -1

    # Find feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    if len(feasible_indices) == 0:
        return -1

    # If only one feasible item, take it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])

    # For each feasible item, do a 1-step lookahead:
    # pick that item, then greedily fill remaining capacity with best density items
    # Score = total value from this simulation

    # Pre-sort all items by value density for fast greedy fill
    densities = remaining_values / remaining_weights
    sorted_indices = np.argsort(-densities)  # descending density

    best_score = -1.0
    best_index = -1

    # Limit lookahead to top candidates for speed
    # Evaluate top K feasible items by density, but also consider some lower-density ones
    feasible_densities = densities[feasible_indices]
    # Sort feasible indices by density descending
    feasible_sorted = feasible_indices[np.argsort(-feasible_densities)]

    # Evaluate a subset for speed: top 20 + some random sampling
    max_eval = min(30, len(feasible_sorted))
    candidates = feasible_sorted[:max_eval]

    for idx in candidates:
        wi = remaining_weights[idx]
        vi = remaining_values[idx]
        cap_after = remaining_capacity - wi

        if cap_after < 0:
            continue

        # Simulate greedy fill after picking idx
        total_value = vi
        remaining_cap = cap_after
        picked = set([idx])

        # Greedily add items by density
        for j in sorted_indices:
            if j in picked:
                continue
            wj = remaining_weights[j]
            if wj > remaining_cap:
                continue
            total_value += remaining_values[j]
            remaining_cap -= wj
            picked.add(j)
            if remaining_cap <= 0:
                break

        if total_value > best_score:
            best_score = total_value
            best_index = int(idx)

    if best_index == -1:
        # Fallback: best density
        return int(np.argmax(densities[feasible_indices]))

    return best_index
# EVOLVE-BLOCK-END
