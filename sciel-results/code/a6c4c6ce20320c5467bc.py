# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 140.0  # 10s safety margin before the 150s kill

def _exact_knapsack(rem_cap, weights, values):
    """Exact 0/1 knapsack value for (rem_cap, weights, values) using integer-unit DP.
    Returns 0.0 if rem_cap <= 0 or no item fits."""
    if rem_cap <= 0:
        return 0.0
    # integer weight units: w_int = int(round(w*1000)); cap_units = int(round(rem_cap*1000))
    scale = 1000.0
    cap_units = int(round(rem_cap * scale))
    if cap_units <= 0:
        return 0.0
    w_int = np.rint(weights * scale).astype(np.int64)
    v = np.asarray(values, dtype=np.float64)
    m = len(w_int)
    if m == 0:
        return 0.0
    # keep only items that could possibly fit
    fit = w_int <= cap_units
    if not np.any(fit):
        return 0.0
    w_int = w_int[fit]
    v = v[fit]
    # DP array length cap_units+1
    dp = np.zeros(cap_units + 1, dtype=np.float64)
    for wi, vi in zip(w_int, v):
        wi = int(wi)
        if wi > cap_units:
            continue
        # dp[wi:] = max(dp[wi:], dp[:-wi] + vi)  (right-to-left in-place)
        dp[wi:] = np.maximum(dp[wi:], dp[:-wi] + vi)
    return float(dp[cap_units])

def _greedy_fill(rem_cap, weights, values):
    """Fast density-greedy fill; return total value."""
    if rem_cap <= 0:
        return 0.0
    mask = weights <= rem_cap
    if not np.any(mask):
        return 0.0
    w = weights[mask]
    v = values[mask]
    densities = v / w
    order = np.argsort(-densities)
    total_val = 0.0
    rem = rem_cap
    for idx in order:
        wi = w[idx]
        vi = v[idx]
        if wi <= rem:
            total_val += vi
            rem -= wi
        if rem <= 0:
            break
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if len(remaining_weights) == 0:
        return -1

    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)

    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    fw = weights[feasible_mask]
    fv = values[feasible_mask]
    n = len(fw)

    # Deadline / size guard: if too big or too late, use density-greedy fill for the lookahead.
    scale = 1000.0
    cap_units = int(round(remaining_capacity * scale))
    use_exact = (
        time.time() < _DEADLINE - 5.0
        and cap_units <= 2000
        and (n * cap_units) <= 4_000_000
    )

    best_score = -1.0
    best_candidate = 0

    for i in range(n):
        rem_cap = remaining_capacity - fw[i]
        if rem_cap < 0:
            continue
        # build remaining arrays excluding i
        if i == 0:
            rw = fw[1:]
            rv = fv[1:]
        elif i == n - 1:
            rw = fw[:-1]
            rv = fv[:-1]
        else:
            rw = np.concatenate((fw[:i], fw[i + 1:]))
            rv = np.concatenate((fv[:i], fv[i + 1:]))

        if use_exact:
            fill = _exact_knapsack(rem_cap, rw, rv)
        else:
            fill = _greedy_fill(rem_cap, rw, rv)

        score = fv[i] + fill
        if score > best_score:
            best_score = score
            best_candidate = i

    # Deadline guard fallback: if we are over, just return the best found (already valid).
    return int(feasible_indices[best_candidate])
# EVOLVE-BLOCK-END
