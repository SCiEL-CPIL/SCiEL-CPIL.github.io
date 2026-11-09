"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np
from collections import deque


def _bottom_level(times, preds):
    """Compute bottom-level (longest path to a sink) for each task.

    times: array of task durations, indexed by task id.
    preds: list where preds[i] is the list of immediate predecessors of task i.
    Returns an array bl where bl[i] = longest path length from i to any sink,
    including i's own duration.
    """
    n = len(times)
    # Build successors list.
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)

    # Topological order (Kahn's algorithm).
    indeg = [0] * n
    for i in range(n):
        for p in preds[i]:
            indeg[i] += 1
    q = deque([i for i in range(n) if indeg[i] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    # If graph has a cycle, fall back to original order.
    if len(topo) != n:
        topo = list(range(n))

    # Compute bl in reverse topological order.
    bl = np.zeros(n, dtype=float)
    for u in reversed(topo):
        best = 0.0
        for v in succs[u]:
            if bl[v] > best:
                best = bl[v]
        bl[u] = times[u] + best
    return bl, succs


def priority(instance):
    t0 = _time.time()
    DEADLINE = 140.0  # safety margin before the 150s kill

    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)

    # Compute bottom-level rank and successor counts.
    bl, succs = _bottom_level(times, preds)
    succ_count = np.array([len(s) for s in succs], dtype=float)
    pred_count = np.array([len(p) for p in preds], dtype=float)

    # Normalize components to [0, 1] for meaningful combination
    bl_max = float(np.max(bl)) if n else 1.0
    bl_min = float(np.min(bl)) if n else 0.0
    bl_range = bl_max - bl_min if bl_max > bl_min else 1.0
    bl_norm = (bl - bl_min) / bl_range

    t_max = float(np.max(times)) if n else 1.0
    t_min = float(np.min(times)) if n else 0.0
    t_range = t_max - t_min if t_max > t_min else 1.0
    t_norm = (times - t_min) / t_range

    sc_max = float(np.max(succ_count)) if n else 1.0
    sc_min = float(np.min(succ_count)) if n else 0.0
    sc_range = sc_max - sc_min if sc_max > sc_min else 1.0
    sc_norm = (succ_count - sc_min) / sc_range

    pc_max = float(np.max(pred_count)) if n else 1.0
    pc_min = float(np.min(pred_count)) if n else 0.0
    pc_range = pc_max - pc_min if pc_max > pc_min else 1.0
    pc_norm = (pred_count - pc_min) / pc_range

    # Weighted combination:
    # - Bottom-level (0.60): primary driver for critical path
    # - Task time (0.20): longer tasks reduce fragmentation
    # - Successor count (0.15): tasks that unlock more future work
    # - Inverse predecessor count (0.05): tasks with fewer deps are more flexible
    score = (0.60 * bl_norm +
             0.20 * t_norm +
             0.15 * sc_norm +
             0.05 * (1.0 - pc_norm))

    # Scale back to original magnitude to preserve relative ordering
    score = score * bl_range + bl_min

    if _time.time() - t0 > DEADLINE:
        # Fallback to longest-task-first.
        return times

    return score
# EVOLVE-BLOCK-END
