"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import deque

_DEADLINE = 140.0  # safety margin before 150s kill


def _compute_priorities(instance):
    """Compute a priority score per task using smooth combined heuristic."""
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = float(instance.get("cycle_time", 40.0))
    preds = instance.get("predecessors", None)

    # Build adjacency: succs[i] = list of successors of task i
    succs = [[] for _ in range(n)]
    if preds is not None:
        for i in range(n):
            p = preds[i]
            if p is not None:
                for j in p:
                    succs[j].append(i)

    # Longest path from each node to any sink (downstream depth in time)
    # Use topological order (Kahn's algorithm)
    in_deg = [0] * n
    for i in range(n):
        if preds is not None and preds[i] is not None:
            in_deg[i] = len(preds[i])

    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    rank = np.zeros(n, dtype=float)  # topological rank for tie-breaking
    rank_idx = 0
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        rank[u] = rank_idx
        rank_idx += 1
        for v in succs[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)

    # If topo order is incomplete (cycle), fall back to LPT
    if len(topo_order) != n:
        return times.copy()

    # down_time[i] = max time from i to any sink (longest path downstream including i)
    down_time = np.zeros(n, dtype=float)
    # Process in reverse topological order
    for u in reversed(topo_order):
        down_time[u] = times[u]
        for v in succs[u]:
            if down_time[v] > 0 or len(succs[u]) > 0:
                down_time[u] = max(down_time[u], times[u] + down_time[v])

    # up_time[i] = max time from any source to i (longest path upstream including i)
    up_time = np.zeros(n, dtype=float)
    for u in topo_order:
        up_time[u] = times[u]
        if preds is not None and preds[u] is not None:
            for v in preds[u]:
                up_time[u] = max(up_time[u], up_time[v] + times[u])

    # Critical path length
    critical_path = float(np.max(up_time)) if n > 0 else 0.0

    # Total work in downstream subtree of task i (including i)
    desc_work = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        desc_work[u] = times[u]
        for v in succs[u]:
            desc_work[u] += desc_work[v]

    # Minimum stations needed for downstream: max of critical path and work-based
    min_stations_down = np.ceil(down_time / cycle_time)
    work_stations = np.ceil(desc_work / cycle_time)
    min_stations = np.maximum(min_stations_down, work_stations)
    min_stations = np.maximum(min_stations, 1.0)

    # Component 1: Bottleneck density (time per minimum station)
    bottleneck_density = times / min_stations

    # Component 2: Critical path proximity (smooth)
    path_length = up_time + down_time - times
    dist_from_critical = critical_path - path_length
    dist_from_critical = np.maximum(dist_from_critical, 0)
    # Use a sharper decay to strongly favor critical-path tasks
    alpha_critical = 1.0
    criticality_score = np.exp(-alpha_critical * dist_from_critical)

    # Component 3: Successor urgency (tasks with many successors are bottlenecks)
    num_succs = np.array([len(s) for s in succs], dtype=float)
    succ_urgency = num_succs / (num_succs + 2.0)  # in [0,1)

    # Component 4: Predecessor flexibility (tasks with many predecessors have more flexibility)
    num_preds = np.array([len(p) if p is not None else 0 for p in preds], dtype=float)
    pred_flexibility = num_preds / (num_preds + 2.0)  # in [0,1)

    # Component 5: Downstream work ratio (how much work depends on this task)
    total_work = float(np.sum(times))
    if total_work > 0:
        downstream_ratio = desc_work / total_work
    else:
        downstream_ratio = np.zeros(n, dtype=float)

    # Component 6: Fit quality - how well the task fits into a single station
    # A task that uses a large fraction of cycle time is harder to pack
    # We want to prioritize tasks that are "tight" (use most of the station)
    fit_ratio = times / cycle_time
    # Smooth penalty: tasks that are too small get lower priority (harder to pack)
    # Use a sigmoid-like function that peaks around 0.5-0.8 of cycle time
    fit_score = 1.0 - np.abs(fit_ratio - 0.6)  # peaks at 0.6, ranges [0, 1.4]
    fit_score = np.clip(fit_score, 0, 1)

    # Component 7: Rank-based tie-breaker (earlier in topo order = higher priority)
    max_rank = float(n - 1) if n > 1 else 1.0
    rank_score = 1.0 - (rank / max_rank)  # 1.0 for first, 0.0 for last

    # Normalize bottleneck density
    max_bd = np.max(bottleneck_density) if n > 0 else 1.0
    if max_bd > 0:
        bd_norm = bottleneck_density / max_bd
    else:
        bd_norm = np.zeros(n, dtype=float)

    # Combine components with weighted sum
    # Higher weight on critical path and bottleneck density
    priority_scores = (
        1.5 * bd_norm +
        1.2 * criticality_score +
        0.5 * succ_urgency +
        0.4 * downstream_ratio +
        0.3 * fit_score +
        0.2 * rank_score -
        0.3 * pred_flexibility
    )

    # Ensure all priorities are positive
    eps = 1e-10
    priority_scores = np.maximum(priority_scores, eps)

    return priority_scores


def priority(instance):
    """Score each assembly task; higher priority tasks are placed first."""
    start = time.time()
    try:
        scores = _compute_priorities(instance)
    except Exception:
        # Fallback to LPT
        try:
            scores = np.asarray(instance["times"], dtype=float)
        except Exception:
            scores = np.zeros(1)

    elapsed = time.time() - start
    if elapsed > _DEADLINE:
        # Shouldn't happen for 30 tasks, but guard anyway
        pass

    return scores
# EVOLVE-BLOCK-END
