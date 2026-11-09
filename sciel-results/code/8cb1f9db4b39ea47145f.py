"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 115.0  # 5s safety margin before 120s limit


def _elapsed_ok():
    return (time.time() - _START_TIME) < _DEADLINE


def _simulate_total(capacity, weights, values, order, skip_indices):
    """Simulate greedy fill following `order`, skipping items in `skip_indices`. Return total value."""
    if isinstance(skip_indices, (int, np.integer)):
        skip_set = {skip_indices}
    else:
        skip_set = set(skip_indices)
    cap = capacity
    val = 0.0
    for i in order:
        if i in skip_set:
            continue
        w = weights[i]
        if w <= cap:
            cap -= w
            val += values[i]
        if cap <= 0:
            break
    return val


def _simulate_2step(capacity, weights, values, order, first_idx):
    """
    2-step lookahead: pick first_idx, then pick the best second item j,
    then continue greedy. Returns max total value over choice of j.
    """
    w1 = weights[first_idx]
    if w1 > capacity:
        return -1.0
    cap_after_1 = capacity - w1
    v1 = values[first_idx]
    if cap_after_1 <= 0:
        return v1

    # For each feasible second item j (j != first_idx), simulate greedy after picking both
    best_total = v1  # fallback: just pick first item, no second
    for j in order:
        if j == first_idx:
            continue
        w2 = weights[j]
        if w2 > cap_after_1:
            continue
        # Simulate greedy fill after picking first_idx and j
        sim_val = v1 + values[j] + _simulate_total(
            cap_after_1 - w2,
            weights,
            values,
            order,
            (first_idx, j)
        )
        if sim_val > best_total:
            best_total = sim_val

    return best_total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # If time is running out, fall back to simple density greedy
    if not _elapsed_ok():
        return int(np.argmax(remaining_values / remaining_weights))

    # Compute density-based order (the baseline greedy order)
    densities = remaining_values / remaining_weights
    order = np.argsort(-densities)  # descending density

    # Determine how many top candidates to evaluate with 2-step lookahead
    # vs 1-step. Use 2-step for top K candidates if time permits.
    # Start with 1-step for all, then upgrade top candidates to 2-step.

    # Phase 1: 1-step lookahead for all feasible items
    candidates = []  # list of (score, idx)
    for i in order:
        w_i = remaining_weights[i]
        if w_i > remaining_capacity:
            continue  # infeasible
        sim_val = remaining_values[i] + _simulate_total(
            remaining_capacity - w_i,
            remaining_weights,
            remaining_values,
            order,
            i
        )
        candidates.append((sim_val, i))

        if not _elapsed_ok():
            break

    if not candidates:
        return -1

    # Sort by 1-step score descending
    candidates.sort(key=lambda x: -x[0])

    # Phase 2: 2-step lookahead for top K candidates (if time permits)
    # K depends on remaining time budget
    elapsed = time.time() - _START_TIME
    remaining_time = _DEADLINE - elapsed

    # Estimate cost: 2-step for one candidate is roughly O(n) for the inner loop
    # plus O(n) for simulation = O(n). 1-step was O(n) per candidate.
    # We already spent ~O(n^2) on phase 1. Let's allow up to 3 top candidates for 2-step.
    K = min(3, len(candidates))

    best_score = -1.0
    best_idx = candidates[0][1]

    for rank in range(K):
        if not _elapsed_ok():
            break
        _, idx = candidates[rank]
        w_i = remaining_weights[idx]
        if w_i > remaining_capacity:
            continue

        # 2-step lookahead
        sim_val = _simulate_2step(
            remaining_capacity,
            remaining_weights,
            remaining_values,
            order,
            idx
        )
        if sim_val > best_score:
            best_score = sim_val
            best_idx = idx

    # If we didn't get to 2-step for the top candidate, use 1-step best
    if best_score < 0:
        best_score, best_idx = candidates[0]

    return int(best_idx)
# EVOLVE-BLOCK-END
