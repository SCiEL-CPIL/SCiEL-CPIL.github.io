"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque
import time


def _build_graph(times, predecessors):
    n = len(times)
    successors = defaultdict(list)
    pred_count = [0] * n
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        pred_count[i] = len(preds)
        for p in preds:
            successors[p].append(i)
    return successors, pred_count


def _compute_bottom_level(n, successors, pred_count, times):
    # Bottom level = longest path from node to any sink (including its own time)
    # Use topological order (Kahn's)
    in_deg = pred_count[:]
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    
    # Reverse topological order to compute bottom level
    bl = [0.0] * n
    for u in reversed(topo_order):
        max_succ = 0.0
        for v in successors[u]:
            max_succ = max(max_succ, bl[v])
        bl[u] = times[u] + max_succ
    return bl, topo_order


def _compute_successor_count(n, successors):
    # Count immediate successors
    return [len(successors[i]) for i in range(n)]


def _compute_min_stations_downstream(n, successors, pred_count, times, cycle_time):
    """
    For each task, estimate the minimum number of additional stations needed
    to complete its downstream subtree (successors and their successors).
    This is approximated by the longest path depth in terms of cycle-time units.
    """
    # Use bottom level as a proxy: bl[u] / cycle_time gives a lower bound on stations
    # for the remaining work if tasks could be perfectly packed.
    # But we want a "bottleneck density" - tasks on critical paths where capacity is scarce.
    # 
    # A better estimate: for each node, the minimum stations needed for its subtree
    # is at least ceil(sum of times in subtree / cycle_time) but also at least the
    # longest path in the subtree divided by cycle_time.
    #
    # For simplicity and speed, use bl[u] / cycle_time as the estimated remaining
    # station count required.
    bl = [0.0] * n
    in_deg = pred_count[:]
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    
    for u in reversed(topo_order):
        max_succ = 0.0
        for v in successors[u]:
            max_succ = max(max_succ, bl[v])
        bl[u] = times[u] + max_succ
    
    # Minimum stations downstream (excluding self) for task u
    # is approximately (bl[u] - times[u]) / cycle_time, but at least 0
    min_stations = [(bl[i] - times[i]) / cycle_time if cycle_time > 0 else 0.0 for i in range(n)]
    # Ensure at least a small positive value to avoid division by zero
    min_stations = [max(ms, 1e-6) for ms in min_stations]
    return min_stations


def priority(instance):
    """
    Score each assembly task using bottleneck density:
    priority[i] = times[i] / min_stations_downstream[i]
    
    This penalizes long tasks that sit on critical paths where station capacity
    is scarce, forcing the greedy filler to reserve tight cycle-time slots for
    high-impact tasks.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Build graph
    successors, pred_count = _build_graph(times, predecessors)
    
    # Compute bottom level
    bl, topo_order = _compute_bottom_level(n, successors, pred_count, times)
    
    # Compute min stations downstream
    min_stations = [0.0] * n
    for u in reversed(topo_order):
        max_succ = 0.0
        for v in successors[u]:
            max_succ = max(max_succ, bl[v])
        # bl[u] already includes times[u], so remaining work after u is bl[u] - times[u]
        min_stations[u] = max((bl[u] - times[u]) / cycle_time, 1e-6) if cycle_time > 0 else 1e-6
    
    # Bottleneck density: time / min_stations_downstream
    # Higher priority = more critical (long task on a tight path)
    priority_scores = np.zeros(n, dtype=float)
    for i in range(n):
        priority_scores[i] = times[i] / min_stations[i]
    
    # Add a small tie-breaker: prefer tasks with more successors (they unlock more work)
    succ_counts = np.array([len(successors[i]) for i in range(n)], dtype=float)
    # Normalize succ_counts to [0, 1] range for a small boost
    max_succ = succ_counts.max() if n > 0 else 1
    if max_succ > 0:
        succ_boost = succ_counts / max_succ
    else:
        succ_boost = np.zeros(n, dtype=float)
    
    # Combine: primary signal is bottleneck density, small boost for successor count
    # Use a weighted sum where bottleneck density dominates
    priority_scores = priority_scores + 0.1 * succ_boost
    
    return priority_scores
# EVOLVE-BLOCK-END
