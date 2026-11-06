"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _topo_order(n, preds):
    """Return a topological order of tasks given predecessor lists."""
    in_deg = np.zeros(n, dtype=int)
    succ = [[] for _ in range(n)]
    for j in range(n):
        for p in preds[j]:
            succ[p].append(j)
            in_deg[j] += 1
    from collections import deque
    q = deque([i for i in range(n) if in_deg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                q.append(v)
    if len(order) != n:
        # fallback: natural order (cycle in graph; shouldn't happen)
        order = list(range(n))
    return order, succ


def _compute_priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)

    order, succ = _topo_order(n, preds)

    # Earliest start (ES) from predecessors: ES[j] = max(ES[p] + t[p])
    ES = np.zeros(n, dtype=float)
    for j in order:
        best = 0.0
        for p in preds[j]:
            cand = ES[p] + times[p]
            if cand > best:
                best = cand
        ES[j] = best

    # Latest finish / start: LF[j] = min(LF[s] - t[s]) over successors, else +inf
    # Use a large horizon = sum of all times as a loose bound for LF.
    horizon = float(times.sum())
    LF = np.full(n, horizon, dtype=float)
    for j in reversed(order):
        best = horizon
        for s in succ[j]:
            cand = LF[s] - times[s]
            if cand < best:
                best = cand
        LF[j] = best
    LS = LF - times  # latest start

    # Transitive successor count (number of tasks that transitively depend on j)
    # Compute via reverse topo propagation of reachability counts (approx by
    # summing over immediate successors' counts plus 1, dedup via bitset for small n).
    # n is 30, so use integer bitsets.
    succ_set = [0] * n
    for j in reversed(order):
        mask = 0
        for s in succ[j]:
            mask |= (1 << s) | succ_set[s]
        succ_set[j] = mask
    succ_count = np.array([bin(m).count("1") for m in succ_set], dtype=float)

    # Slack = LS - ES (small slack => critical => schedule early)
    slack = LS - ES
    # Normalize slack to [0,1] range for weighting
    slack_min, slack_max = float(slack.min()), float(slack.max())
    span = max(slack_max - slack_min, 1e-9)
    slack_norm = (slack - slack_min) / span  # 0 = most critical

    # Duration normalization
    t_min, t_max = float(times.min()), float(times.max())
    t_span = max(t_max - t_min, 1e-9)
    t_norm = (times - t_min) / t_span  # 1 = longest

    # Composite: duration dominates, then criticality (inverse slack), then successor count
    sc_min, sc_max = float(succ_count.min()), float(succ_count.max())
    sc_span = max(sc_max - sc_min, 1e-9)
    sc_norm = (succ_count - sc_min) / sc_span

    score = 1.0 * t_norm + 0.6 * (1.0 - slack_norm) + 0.3 * sc_norm

    # Add a tiny unique tie-breaker so equal scores have deterministic distinct order
    eps = 1e-6
    score = score + eps * (np.arange(n) / max(n, 1))

    return score


def priority(instance):
    start = time.time()
    deadline = start + 110.0  # safety margin well under 120 s / 150 s kill
    try:
        res = _compute_priority(instance)
        if time.time() > deadline:
            # fallback to a simple, fast rule
            return np.asarray(instance["times"], dtype=float)
        return res
    except Exception:
        # Robust fallback: longest-task-first
        return np.asarray(instance["times"], dtype=float)
# EVOLVE-BLOCK-END
