"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np
from collections import deque


def _compute_ranks(times, preds):
    """Compute bottom-level (RPV), top-level, and successor counts."""
    n = len(times)
    succs = [[] for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
            indeg[i] += 1

    # Topological sort (Kahn's)
    q = deque([i for i in range(n) if indeg[i] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(topo) != n:
        topo = list(range(n))

    # Bottom-level (RPV): longest path from node to a sink, including node's time
    rpv = np.zeros(n, dtype=float)
    for u in reversed(topo):
        best = 0.0
        for v in succs[u]:
            if rpv[v] > best:
                best = rpv[v]
        rpv[u] = times[u] + best

    # Top-level: longest path from a source to node, including node's time
    top = np.zeros(n, dtype=float)
    for u in topo:
        best = 0.0
        for p in preds[u]:
            if top[p] > best:
                best = top[p]
        top[u] = times[u] + best

    succ_count = np.array([len(s) for s in succs], dtype=float)
    return rpv, top, succ_count


def priority(instance):
    t0 = _time.time()
    DEADLINE = 140.0  # safety margin before the 150s kill

    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)

    rpv, top, succ_count = _compute_ranks(times, preds)

    # Combine: RPV dominates (critical path urgency), then task time (LPT tie-break),
    # then successor count (unlocking more tasks reduces idle time).
    # Scale terms so that RPV differences are primary.
    max_rpv = float(np.max(rpv)) if n else 1.0
    max_time = float(np.max(times)) if n else 1.0

    # Weighting: RPV is primary. Time and succ_count are secondary tie-breakers.
    # Use small weights relative to typical RPV differences.
    w_time = 0.01 * max_time / max(max_rpv, 1.0)
    w_succ = 0.001 * max_rpv / max(max_rpv, 1.0)

    scores = rpv + w_time * times + w_succ * succ_count

    if _time.time() - t0 > DEADLINE:
        # Fallback to longest-task-first.
        return times

    return scores
# EVOLVE-BLOCK-END
