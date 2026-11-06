"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.monotonic()
_TOTAL_BUDGET = 110.0


def _estimate_greedy_completion(capacity, weights, values):
    """Estimate total value achievable by greedy density completion."""
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
        else:
            break
    return total_val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select next item using lookahead: for each feasible candidate,
    estimate the total value if we pick it now plus greedy completion.
    Pick the candidate with the highest estimated total.
    """
    n = len(remaining_weights)
    if n == 0:
        return -1

    if time.monotonic() - _START > _TOTAL_BUDGET:
        return -1

    if remaining_capacity <= 0:
        return -1

    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]

    # For small n, do full lookahead; for large n, use a blend score
    if n <= 30:
        best_score = -1.0
        best_idx = -1
        for idx in feasible_indices:
            w_i = remaining_weights[idx]
            v_i = remaining_values[idx]
            # If we pick idx, what's the greedy completion?
            new_cap = remaining_capacity - w_i
            # Build arrays without idx
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            comp_val = _estimate_greedy_completion(new_cap, remaining_weights[mask], remaining_values[mask])
            total_est = v_i + comp_val
            if total_est > best_score:
                best_score = total_est
                best_idx = idx
        return int(best_idx)
    else:
        # For large n, use a blended score: density + small weight preference
        # This approximates lookahead without O(n^2) cost
        f_w = remaining_weights[feasible_mask]
        f_v = remaining_values[feasible_mask]
        densities = f_v / f_w
        # Blend: primarily density, with slight preference for lighter items
        # to preserve capacity for future items
        norm_w = f_w / (np.max(f_w) + 1e-12)
        scores = densities * (1.0 - 0.1 * norm_w)
        best_local = int(np.argmax(scores))
        return int(feasible_indices[best_local])
# EVOLVE-BLOCK-END
