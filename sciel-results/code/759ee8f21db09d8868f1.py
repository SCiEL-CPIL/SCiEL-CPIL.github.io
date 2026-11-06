"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.monotonic()
_TOTAL_BUDGET = 130.0  # safety margin before 150s kill


def _greedy_fill_fast(capacity, weights, values, excluded_idx):
    """Fast greedy fill with multiple policies, returns best total value."""
    if capacity <= 0:
        return 0.0
    n = len(weights)
    mask = np.ones(n, dtype=bool)
    mask[excluded_idx] = False
    mask &= weights <= capacity
    if not np.any(mask):
        return 0.0

    idxs = np.where(mask)[0]
    w = weights[idxs]
    v = values[idxs]
    fn = len(w)

    best_total = 0.0

    def _run_greedy(order):
        nonlocal best_total
        total = 0.0
        cap = capacity
        for i in order:
            wi = w[i]
            if wi <= cap:
                total += v[i]
                cap -= wi
                if cap <= 0:
                    break
        if total > best_total:
            best_total = total

    # Policy 0: density
    order = np.argsort(-(v / w))
    _run_greedy(order)

    # Policy 1: value
    order = np.argsort(-v)
    _run_greedy(order)

    # Policy 2: blend density + value
    order = np.argsort(-(v / w + 0.5 * v))
    _run_greedy(order)

    # Policy 3: lighter items first
    order = np.argsort(w)
    _run_greedy(order)

    # Policy 4: value - weight penalty
    score4 = v - 0.3 * w
    order = np.argsort(-score4)
    _run_greedy(order)

    # Policy 5: value^2 / weight (emphasizes high value more)
    score5 = v * v / w
    order = np.argsort(-score5)
    _run_greedy(order)

    # Policy 6: (value/weight)^2 - emphasizes density more
    score6 = (v / w) ** 2
    order = np.argsort(-score6)
    _run_greedy(order)

    # Policy 7: value / (weight + 0.1) - damped weight penalty
    score7 = v / (w + 0.1)
    order = np.argsort(-score7)
    _run_greedy(order)

    # Policy 8: geometric mean of normalized value and density
    v_min, v_max = v.min(), v.max()
    d = v / w
    d_min, d_max = d.min(), d.max()
    v_norm = (v - v_min) / (v_max - v_min + 1e-12)
    d_norm = (d - d_min) / (d_max - d_min + 1e-12)
    score8 = np.sqrt(v_norm * d_norm + 1e-12)
    order = np.argsort(-score8)
    _run_greedy(order)

    return best_total


def _two_step_lookahead(capacity, weights, values, first_idx, top_k=5):
    """
    For a given first item, try top-k second items and greedy-fill the rest.
    Returns the best total value achievable with first_idx as the first pick.
    """
    if capacity <= 0:
        return 0.0
    n = len(weights)
    w1 = weights[first_idx]
    v1 = values[first_idx]
    rem_cap1 = capacity - w1
    if rem_cap1 < 0:
        return 0.0

    # Get greedy fill after taking first_idx (baseline)
    base_total = v1 + _greedy_fill_fast(rem_cap1, weights, values, first_idx)

    # Try adding a second item before greedy fill
    feasible_mask = np.ones(n, dtype=bool)
    feasible_mask[first_idx] = False
    feasible_mask &= weights <= rem_cap1
    if not np.any(feasible_mask):
        return base_total

    candidate_indices = np.where(feasible_mask)[0]
    cand_w = weights[candidate_indices]
    cand_v = values[candidate_indices]
    cand_density = cand_v / cand_w
    top_k_actual = min(top_k, len(candidate_indices))
    top_order = np.argsort(-cand_density)[:top_k_actual]

    best_total = base_total
    for k in top_order:
        second_idx = candidate_indices[k]
        w2 = weights[second_idx]
        v2 = values[second_idx]
        rem_cap2 = rem_cap1 - w2
        if rem_cap2 < 0:
            continue
        # Exclude both first and second from greedy fill
        # We need a version of greedy that excludes two indices
        # For simplicity, use the existing one with a trick: temporarily mark
        # Actually, let's just do a simple greedy inline excluding both
        mask = np.ones(n, dtype=bool)
        mask[first_idx] = False
        mask[second_idx] = False
        mask &= weights <= rem_cap2
        if not np.any(mask):
            est = v1 + v2
        else:
            idxs = np.where(mask)[0]
            w_rem = weights[idxs]
            v_rem = values[idxs]
            # Quick density greedy
            order = np.argsort(-(v_rem / w_rem))
            total_rem = 0.0
            cap = rem_cap2
            for i in order:
                wi = w_rem[i]
                if wi <= cap:
                    total_rem += v_rem[i]
                    cap -= wi
                    if cap <= 0:
                        break
            est = v1 + v2 + total_rem
        if est > best_total:
            best_total = est

    return best_total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1

    elapsed = time.monotonic() - _START
    if elapsed > _TOTAL_BUDGET:
        return -1

    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    fw = remaining_weights[feasible_indices]
    fv = remaining_values[feasible_indices]
    fn = len(fw)

    if fn == 1:
        return int(feasible_indices[0])

    # Fast fallback if time is tight: use a combined score
    if elapsed > _TOTAL_BUDGET * 0.75:
        density = fv / fw
        d_min, d_max = density.min(), density.max()
        v_min, v_max = fv.min(), fv.max()
        d_range = d_max - d_min + 1e-12
        v_range = v_max - v_min + 1e-12
        d_norm = (density - d_min) / d_range
        v_norm = (fv - v_min) / v_range
        scores = 0.5 * d_norm + 0.3 * v_norm + 0.2 * (1.0 - (fw - fw.min()) / (fw.max() - fw.min() + 1e-12))
        return int(feasible_indices[int(np.argmax(scores))])

    # Determine how many candidates to fully evaluate based on time
    time_left = _TOTAL_BUDGET - elapsed
    if time_left > 60.0:
        # Full: evaluate all feasible items with 1-step lookahead
        candidates = list(range(fn))
        use_two_step = True
    elif time_left > 30.0:
        # Moderate: evaluate top 20 by density
        density = fv / fw
        top_order = np.argsort(-density)
        candidates = list(top_order[:min(20, fn)])
        use_two_step = True
    else:
        # Quick: evaluate top 10 by density
        density = fv / fw
        top_order = np.argsort(-density)
        candidates = list(top_order[:min(10, fn)])
        use_two_step = False

    best_score = -1.0
    best_local_idx = candidates[0] if candidates else 0

    for li in candidates:
        item_w = fw[li]
        item_v = fv[li]
        remaining_cap = remaining_capacity - item_w
        if remaining_cap < 0:
            continue
        est = _greedy_fill_fast(remaining_cap, remaining_weights, remaining_values,
                                feasible_indices[li])
        total_est = item_v + est

        if use_two_step and remaining_cap > 0:
            # Try 2-step lookahead for this candidate
            two_step_val = _two_step_lookahead(
                remaining_capacity, remaining_weights, remaining_values,
                feasible_indices[li], top_k=3
            )
            if two_step_val > total_est:
                total_est = two_step_val

        if total_est > best_score:
            best_score = total_est
            best_local_idx = li

        # Time check inside loop
        if time.monotonic() - _START > _TOTAL_BUDGET * 0.9:
            break

    return int(feasible_indices[best_local_idx])
# EVOLVE-BLOCK-END
