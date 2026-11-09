"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np
from collections import deque


def _compute_ranks(times, preds):
    """Compute bottom-level (RPV), top-level, successor counts, weighted successor RPV,
    and slack-based critical path density."""
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

    # Weighted successor RPV: sum of RPV of immediate successors
    w_succ_rpv = np.zeros(n, dtype=float)
    for u in range(n):
        if succs[u]:
            w_succ_rpv[u] = float(np.sum(rpv[succs[u]]))

    return rpv, top, succ_count, w_succ_rpv


def priority(instance):
    t0 = _time.time()
    DEADLINE = 140.0  # safety margin before the 150s kill

    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)

    rpv, top, succ_count, w_succ_rpv = _compute_ranks(times, preds)

    ct = float(instance.get("cycle_time", 40.0))

    # Critical Path Density: time / slack
    # slack = cycle_time - rpv[u]  (how much "room" remains after this task's downstream path)
    # Lower slack means the task is more critical to place early.
    # Higher time relative to slack means the task is "denser" and harder to fit.
    # We want HIGH priority for high density (large time, small slack).

    slack = ct - rpv
    # Clamp slack to a small positive floor to avoid division by zero / negative
    eps = 0.01
    slack_safe = np.maximum(slack, eps)

    # Critical path density: ratio of processing time to available slack
    cpd = times / slack_safe

    # Also compute a "tightness" component: how close the task's downstream path is to CT
    # This captures tasks where even a small delay causes overflow
    tightness = np.maximum(1.0 - rpv / ct, 0.0) if ct > 0 else np.ones(n)

    # Normalize components
    max_cpd = float(np.max(cpd)) if n and np.max(cpd) > 0 else 1.0
    max_rpv = float(np.max(rpv)) if n else 1.0
    max_time = float(np.max(times)) if n else 1.0
    max_wsr = float(np.max(w_succ_rpv)) if n and np.max(w_succ_rpv) > 0 else 1.0

    cpd_norm = cpd / max_cpd if max_cpd > 0 else cpd
    rpv_norm = rpv / max_rpv if max_rpv > 0 else rpv
    wsr_norm = w_succ_rpv / max_wsr if max_wsr > 0 else w_succ_rpv
    time_norm = times / max_time if max_time > 0 else times
    succ_norm = succ_count / (np.max(succ_count) if np.max(succ_count) > 0 else 1.0)

    # Weighted combination:
    # CPD (critical path density) is the primary signal - directly encodes
    # "least flexibility relative to duration"
    # RPV provides secondary critical-path urgency
    # Weighted successor RPV captures unlock width
    # Time as LPT tie-breaker
    w_cpd = 1.0
    w_rpv = 0.4
    w_wsr = 0.2
    w_time = 0.1
    w_succ = 0.05

    scores = w_cpd * cpd_norm + w_rpv * rpv_norm + w_wsr * wsr_norm + w_time * time_norm + w_succ * succ_norm

    # Scale to a stable magnitude
    scores = scores * max_rpv

    if _time.time() - t0 > DEADLINE:
        return times

    return scores
# EVOLVE-BLOCK-END
