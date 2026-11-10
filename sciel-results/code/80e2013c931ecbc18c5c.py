"""0/1 knapsack heuristic: lookahead over top-k density candidates with greedy completion."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 150.0  # hard wall-clock guard (seconds)
_start_time = time.perf_counter()
_EPS = 1e-12


def _greedy_complete(remaining_capacity, remaining_weights, remaining_values):
    """Estimate total value by greedy density fill (fast, vectorized sort once)."""
    if remaining_capacity <= _EPS:
        return 0.0
    n = len(remaining_weights)
    if n == 0:
        return 0.0
    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)
    # Only feasible items (weight <= capacity)
    feas = w <= remaining_capacity + _EPS
    if not np.any(feas):
        return 0.0
    wf = w[feas]
    vf = v[feas]
    # density = value / weight; guard against zero weight
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = np.where(wf > _EPS, vf / np.maximum(wf, _EPS), np.inf)
    order = np.argsort(-dens, kind="stable")
    total = 0.0
    cap = remaining_capacity
    for idx in order:
        wi = wf[idx]
        vi = vf[idx]
        if wi <= cap + _EPS:
            total += vi
            cap -= wi
            if cap <= _EPS:
                break
    return total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Deadline guard: if we are too close to the limit, fall back to the
    # cheapest safe choice (highest density) to avoid a timeout.
    if time.perf_counter() - _start_time > _DEADLINE - 5.0:
        if len(remaining_weights) == 0 or remaining_capacity <= _EPS:
            return -1
        w = np.asarray(remaining_weights, dtype=np.float64)
        v = np.asarray(remaining_values, dtype=np.float64)
        feas = w <= remaining_capacity + _EPS
        if not np.any(feas):
            return -1
        wf = w[feas]
        vf = v[feas]
        with np.errstate(divide="ignore", invalid="ignore"):
            dens = np.where(wf > _EPS, vf / np.maximum(wf, _EPS), np.inf)
        return int(np.argwhere(feas)[int(np.argmax(dens))][0])

    if len(remaining_weights) == 0 or remaining_capacity <= _EPS:
        return -1

    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)
    n = len(w)

    feas = w <= remaining_capacity + _EPS
    if not np.any(feas):
        return -1

    # Candidate set: top-k by density among feasible items, plus a value tie-break.
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = np.where(w > _EPS, v / np.maximum(w, _EPS), np.inf)
    dens = np.where(feas, dens, -np.inf)

    k = min(n, 12)
    # top-k indices by density (stable), restricted to feasible
    topk = np.argsort(-dens, kind="stable")[:k]

    best_idx = -1
    best_score = -np.inf
    # For each candidate, estimate completion value
    for cand in topk:
        if not feas[cand]:
            continue
        # new capacity after taking cand
        new_cap = remaining_capacity - w[cand]
        if new_cap < -_EPS:
            continue
        # remaining items excluding cand
        mask = np.ones(n, dtype=bool)
        mask[cand] = False
        rem_w = w[mask]
        rem_v = v[mask]
        # estimated total = v[cand] + greedy completion
        est = v[cand] + _greedy_complete(max(new_cap, 0.0), rem_w, rem_v)
        # tie-break: prefer higher raw value, then lower weight
        if est > best_score + _EPS or (
            abs(est - best_score) <= _EPS and v[cand] > v[best_idx]
        ):
            best_score = est
            best_idx = int(cand)

    if best_idx == -1:
        # Fallback: best density feasible
        best_idx = int(np.argwhere(feas)[int(np.argmax(dens))][0])
    return int(best_idx)
# EVOLVE-BLOCK-END
