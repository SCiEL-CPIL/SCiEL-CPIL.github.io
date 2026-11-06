"""Bounded lookahead 0/1 knapsack heuristic with adaptive deep greedy estimation."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.monotonic()
_TOTAL_BUDGET = 118.0  # safety margin under the 120 s per-eval / 150 s kill

def _remaining_budget():
    return max(0.0, _TOTAL_BUDGET - (time.monotonic() - _START))

def _greedy_complete(w, v, cap):
    """
    Fast greedy completion with a 1-step lookahead to beat pure density.
    Returns estimated total added value.
    """
    n = len(w)
    if n == 0 or cap <= 0:
        return 0.0

    # Quick pure-density pass as baseline
    safe_w = np.maximum(w, 1e-12)
    order = np.argsort(-v / safe_w)
    total = 0.0
    c = float(cap)
    used = np.zeros(n, dtype=bool)
    for i in order:
        wi = w[i]
        if wi <= c + 1e-12:
            total += v[i]
            c -= wi
            used[i] = True
            if c <= 1e-9:
                break
    base = total

    # 1-step lookahead: try swapping the first pick with a slightly different one
    # Only do this when n is small enough to be cheap
    if n > 60 or cap <= 0:
        return base

    # Find the best single-item start, then complete
    # Top-3 by density and top-3 by value as starts
    starts = set()
    top_d = np.argsort(-v / safe_w)[:min(3, n)]
    top_v = np.argsort(-v)[:min(3, n)]
    for idx in top_d:
        starts.add(int(idx))
    for idx in top_v:
        starts.add(int(idx))

    best = base
    for si in starts:
        sw = float(w[si])
        if sw > cap + 1e-12:
            continue
        sv = float(v[si])
        rem_cap = cap - sw
        # Complete the rest with pure density
        mask = np.ones(n, dtype=bool)
        mask[si] = False
        rw = w[mask]
        rv = v[mask]
        rn = len(rw)
        if rn == 0:
            est = sv
        else:
            safe_rw = np.maximum(rw, 1e-12)
            rorder = np.argsort(-rv / safe_rw)
            rc = rem_cap
            rt = 0.0
            for ri in rorder:
                rwi = rw[ri]
                if rwi <= rc + 1e-12:
                    rt += rv[ri]
                    rc -= rwi
                    if rc <= 1e-9:
                        break
            est = sv + rt
        if est > best:
            best = est

    return best

def _score_candidates(w, v, cap, n):
    """Generate diverse candidate indices using multiple scoring families."""
    safe_w = np.maximum(w, 1e-12)
    density = v / safe_w
    rel_w = w / float(cap) if cap > 0 else np.zeros(n)

    scores = np.vstack([
        v,                                        # 1. pure value
        density,                                  # 2. pure density
        density + 0.1 * v,                        # 3. blend
        v - 0.02 * w,                             # 4. value minus weight penalty
        density * (1.0 + 0.3 * rel_w),            # 5. density boosted by relative size
        v / (w + 0.1 * cap),                      # 6. smoothed density
        density - 0.2 * np.abs(rel_w - 0.25),     # 7. density with size-shape term
        v * np.exp(-w / (cap + 1e-9)),            # 8. value decaying with weight
        v * (1.0 - rel_w * 0.5),                  # 9. value discounted by fraction of cap
        density + 0.05 * v * (1.0 - rel_w),       # 10. density + value for smaller items
    ])

    cands = set()
    k = min(n, 12)
    for s in scores:
        top = np.argsort(-s)[:k]
        for t in top:
            cands.add(int(t))

    # Ensure density-argmax and value-argmax are included
    cands.add(int(np.argmax(density)))
    cands.add(int(np.argmax(v)))

    return list(cands), density

def _evaluate_candidate(ci, w, v, cap, n):
    """
    Estimate total value if we pick candidate ci now, using greedy completion
    with 1-step lookahead on the remainder.
    """
    used_w = float(w[ci])
    used_v = float(v[ci])
    new_cap = cap - used_w
    if new_cap < -1e-9:
        return -np.inf

    # Build arrays excluding candidate
    mask = np.ones(n, dtype=bool)
    mask[ci] = False
    rw = w[mask]
    rv = v[mask]
    rn = len(rw)

    if rn == 0:
        return used_v

    return used_v + _greedy_complete(rw, rv, new_cap)

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    cap = float(remaining_capacity)
    if cap <= 0:
        return -1

    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)
    n = len(w)
    if n == 0:
        return -1

    # Only consider feasible items
    feasible = w <= cap + 1e-12
    if not np.any(feasible):
        return -1

    # Fast path if near deadline
    if _remaining_budget() < 0.1:
        feasible_v = np.where(feasible, v / np.maximum(w, 1e-12), -np.inf)
        idx = int(np.argmax(feasible_v))
        return idx if feasible[idx] else -1

    # Generate diverse candidates
    cands, density = _score_candidates(w, v, cap, n)

    # Filter to feasible candidates only
    cands = [c for c in cands if feasible[c]]
    if not cands:
        return -1

    # Adaptive candidate limit: fewer candidates when many items remain (early),
    # more when few remain (late) because each evaluation is cheaper.
    if n > 80:
        sim_limit = 25
    elif n > 40:
        sim_limit = 35
    else:
        sim_limit = 50

    if len(cands) > sim_limit:
        cands.sort(key=lambda i: -density[i])
        cands = cands[:sim_limit]

    # Evaluate each candidate
    best_idx = cands[0]
    best_est = -np.inf
    ests = []

    for ci in cands:
        est = _evaluate_candidate(ci, w, v, cap, n)
        ests.append((est, ci))
        if est > best_est:
            best_est = est
            best_idx = ci

        # Deadline check
        if _remaining_budget() < 0.5:
            break

    # Soft-max tie-breaking: if top estimates are close, prefer the one with
    # higher density as a secondary signal (reduces arbitrary order effects).
    if len(ests) >= 2:
        ests.sort(key=lambda x: -x[0])
        top_est = ests[0][0]
        # Consider candidates within 1% of the best estimate
        close = [(e, c) for e, c in ests if e >= top_est * 0.99]
        if len(close) > 1:
            # Among close candidates, pick the one with highest density
            close.sort(key=lambda x: -density[x[1]])
            best_idx = close[0][1]
            best_est = close[0][0]

    return int(best_idx)
# EVOLVE-BLOCK-END
