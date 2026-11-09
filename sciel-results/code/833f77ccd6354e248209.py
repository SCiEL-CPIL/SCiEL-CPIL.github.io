"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

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

    from collections import deque
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
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

    # Minimum number of remaining stations needed for downstream subtree of task i
    desc_work = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        desc_work[u] = times[u]
        for v in succs[u]:
            desc_work[u] += desc_work[v]

    # Minimum stations needed for downstream
    min_stations_down = np.ceil(down_time / cycle_time)
    work_stations = np.ceil(desc_work / cycle_time)
    min_stations = np.maximum(min_stations_down, work_stations)
    min_stations = np.maximum(min_stations, 1)

    # Component 1: Bottleneck density (time per minimum station)
    bottleneck_density = times / min_stations

    # Component 2: Critical path proximity (smooth)
    # Use soft-max to measure how close each task is to the critical path
    # criticality_score[i] = exp(-alpha * (critical_path - (up_time[i] + down_time[i] - times[i])))
    # This gives higher scores to tasks closer to the critical path
    path_length = up_time + down_time - times
    # Normalize: distance from critical path
    dist_from_critical = critical_path - path_length
    dist_from_critical = np.maximum(dist_from_critical, 0)
    # Soft criticality: use exponential decay
    alpha_critical = 0.5
    criticality_score = np.exp(-alpha_critical * dist_from_critical)

    # Component 3: Successor urgency (tasks with many successors are bottlenecks)
    num_succs = np.array([len(s) for s in succs], dtype=float)
    # Use a smooth saturation function
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

    # Combine using log-sum-exp for smooth maximum-like behavior
    # We want to combine these signals multiplicatively/additively in a smooth way
    # Use weighted geometric mean as a smooth surrogate for min/max

    # Normalize each component to [0, 1] range
    max_bd = np.max(bottleneck_density) if n > 0 else 1.0
    if max_bd > 0:
        bd_norm = bottleneck_density / max_bd
    else:
        bd_norm = np.zeros(n, dtype=float)

    # Weights for each component
    w_bd = 1.0       # bottleneck density
    w_crit = 0.8     # critical path proximity
    w_succ = 0.6     # successor urgency
    w_pred = -0.3    # predecessor flexibility (negative: more preds = lower priority)
    w_down = 0.4     # downstream work ratio

    # Compute weighted sum of log-components for geometric mean
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    log_terms = np.log(np.maximum(bd_norm, eps) * w_bd + 
                        criticality_score * w_crit + 
                        succ_urgency * w_succ + 
                        (1 - pred_flexibility) * abs(w_pred) + 
                        downstream_ratio * w_down + eps)
    
    # Actually, let's use a simpler but effective combination:
    # Priority = weighted sum with smooth interactions
    priority_scores = (
        1.2 * bottleneck_density +
        0.8 * criticality_score * np.max(bottleneck_density) +
        0.5 * succ_urgency * np.max(bottleneck_density) +
        0.3 * downstream_ratio * np.max(bottleneck_density) -
        0.2 * pred_flexibility * np.max(bottleneck_density)
    )

    # Ensure all priorities are positive
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
