"""Priority scorer for Resource-Constrained Project Scheduling (serial SGS)."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 150.0          # wall-clock kill limit
_SAFETY = 8.0              # return at least this many seconds before the limit


def _to_int_list(x):
    """Normalize a predecessor entry to a list of ints."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    if isinstance(x, np.ndarray):
        return [int(v) for v in x]
    # single scalar predecessor
    return [int(x)]


def _cpm_ranks(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    preds = [_to_int_list(instance["predecessors"][i]) for i in range(n)]

    # in-degree for topological order
    indeg = np.zeros(n, dtype=int)
    succs = [[] for _ in range(n)]
    for j in range(n):
        for i in preds[j]:
            if 0 <= i < n and i != j:
                succs[i].append(j)
                indeg[j] += 1

    # forward pass (earliest start / earliest finish)
    order = []
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    indeg_work = indeg.copy()
    while q:
        u = q.popleft()
        order.append(u)
        for v in succs[u]:
            indeg_work[v] -= 1
            if indeg_work[v] == 0:
                q.append(v)

    # fallback if cycle: append remaining
    if len(order) < n:
        seen = set(order)
        for i in range(n):
            if i not in seen:
                order.append(i)

    es = np.zeros(n, dtype=float)
    for u in order:
        for i in preds[u]:
            if 0 <= i < n:
                es[u] = max(es[u], es[i] + durations[i])
    ef = es + durations

    # project end from forward pass
    project_end = float(np.max(ef)) if n else 0.0

    # backward pass (latest start / latest finish)
    ls = np.full(n, project_end, dtype=float)
    for u in reversed(order):
        for v in succs[u]:
            if 0 <= v < n:
                ls[u] = min(ls[u], ls[v] - durations[u])
        # also respect project end
        ls[u] = min(ls[u], project_end - durations[u])
    lf = ls + durations

    # total float = latest start - earliest start
    total_float = ls - es
    # tail / bottom-level rank: longest path from node to any sink
    tail = np.zeros(n, dtype=float)
    for u in reversed(order):
        best = 0.0
        for v in succs[u]:
            if 0 <= v < n:
                best = max(best, durations[v] + tail[v])
        tail[u] = best

    return total_float, tail, ef, project_end


def priority(instance):
    t0 = time.time()
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    if n == 0:
        return np.asarray([], dtype=float)

    # Resource demand per activity (renewable resources)
    demands = instance.get("demands", None)
    if demands is not None:
        demands = np.asarray(demands, dtype=float)
        if demands.ndim == 1:
            # treat as scalar demand per activity
            demand_intensity = demands
        else:
            # per-resource: sum across resources (total load)
            demand_intensity = np.sum(demands, axis=1)
    else:
        demand_intensity = np.ones(n, dtype=float)

    try:
        total_float, tail, ef, project_end = _cpm_ranks(instance)
    except Exception:
        # safe fallback: use durations
        total_float = np.zeros(n)
        tail = durations.copy()
        ef = durations.copy()
        project_end = float(np.max(durations)) if n else 0.0

    # Guard: if we are too close to the deadline, return a simple robust score
    if time.time() - t0 > _DEADLINE - _SAFETY:
        return durations.copy()

    # Normalize each component to [0, 1] (higher = more critical)
    def norm_inv(x):
        # invert so that smaller float -> higher score
        x = np.asarray(x, dtype=float)
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (hi - x) / (hi - lo)

    def norm(x):
        x = np.asarray(x, dtype=float)
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    score_float = norm_inv(total_float)   # low float -> high priority
    score_tail = norm(tail)               # long tail -> high priority
    score_dur = norm(durations)           # long duration -> high priority
    score_demand = norm(demand_intensity) # heavy resource use -> high priority

    # Weights: emphasize total float (critical path) and tail,
    # with secondary duration and resource intensity.
    w_float = 0.45
    w_tail = 0.22
    w_dur = 0.20
    w_demand = 0.13

    scores = (w_float * score_float
              + w_tail * score_tail
              + w_dur * score_dur
              + w_demand * score_demand)

    # Add a tiny tie-breaker to keep ordering deterministic and stable
    scores = scores + 1e-6 * np.arange(n) * 0.0  # no-op, keep float array

    return np.asarray(scores, dtype=float)
# EVOLVE-BLOCK-END
