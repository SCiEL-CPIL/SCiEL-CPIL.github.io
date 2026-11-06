"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.monotonic()
_TOTAL_BUDGET = 110.0


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

    best_total = 0.0

    # Policy 0: density
    order = np.argsort(-(v / w))
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

    # Policy 1: value
    order = np.argsort(-v)
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

    # Policy 2: blend (density + 0.5 * value)
    order = np.argsort(-(v / w + 0.5 * v))
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

    # Policy 3: small-weight first (to pack more items)
    order = np.argsort(w)
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

    # Policy 4: high value-to-weight with preference for items that fit snugly
    # (items whose weight is a good fraction of remaining capacity)
    if cap > 0 and len(w) > 0:
        # Score: density * (1 + 0.1 * (w / capacity))
        scores = (v / w) * (1 + 0.1 * (w / capacity))
        order = np.argsort(-scores)
        total = 0.0
        cap2 = capacity
        for i in order:
            wi = w[i]
            if wi <= cap2:
                total += v[i]
                cap2 -= wi
                if cap2 <= 0:
                    break
        if total > best_total:
            best_total = total

    return best_total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """Select next item using lookahead greedy completion with multiple policies."""
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

    # Fast fallback if time is tight
    if elapsed > _TOTAL_BUDGET * 0.7:
        density = fv / fw
        d_min, d_max = density.min(), density.max()
        v_min, v_max = fv.min(), fv.max()
        d_range = d_max - d_min + 1e-12
        v_range = v_max - v_min + 1e-12
        d_norm = (density - d_min) / d_range
        v_norm = (fv - v_min) / v_range
        scores = 0.6 * d_norm + 0.4 * v_norm
        return int(feasible_indices[int(np.argmax(scores))])

    # Lookahead: for each feasible item, estimate total via greedy completion
    best_score = -1.0
    best_local_idx = 0

    for li in range(fn):
        item_w = fw[li]
        item_v = fv[li]
        remaining_cap = remaining_capacity - item_w
        est = _greedy_fill_fast(remaining_cap, remaining_weights, remaining_values,
                                feasible_indices[li])
        total_est = item_v + est
        if total_est > best_score:
            best_score = total_est
            best_local_idx = li

    return int(feasible_indices[best_local_idx])
# EVOLVE-BLOCK-END
