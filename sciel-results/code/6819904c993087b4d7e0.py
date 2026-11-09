"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import heapq
import time


def _build_graph(instance):
    """Build adjacency and in-degree arrays from the instance."""
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)

    adj = [[] for _ in range(n)]
    in_deg = np.zeros(n, dtype=int)
    succ_count = np.zeros(n, dtype=int)

    for i in range(n):
        for p in preds[i]:
            adj[p].append(i)
            in_deg[i] += 1
            succ_count[p] += 1

    return times, adj, in_deg, succ_count


def _topological_sort(adj, in_deg, n):
    """Return a topological ordering of tasks."""
    queue = deque()
    for i in range(n):
        if in_deg[i] == 0:
            queue.append(i)
    
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adj[node]:
            in_deg[neighbor] -= 1
            if in_deg[neighbor] == 0:
                queue.append(neighbor)
    
    return order


def _simulate(instance, priority_scores):
    """
    Simulate the greedy list scheduling with given priority scores.
    Returns number of stations.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    adj = [[] for _ in range(n)]
    in_deg = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            adj[p].append(i)
            in_deg[i] += 1
    
    completed = [False] * n
    station_load = 0.0
    num_stations = 1
    remaining_in_deg = in_deg.copy()
    
    while True:
        eligible = []
        for i in range(n):
            if not completed[i] and remaining_in_deg[i] == 0:
                eligible.append(i)
        
        if not eligible:
            break
        
        eligible.sort(key=lambda i: priority_scores[i], reverse=True)
        
        placed_any = False
        for i in eligible:
            if station_load + times[i] <= cycle_time:
                completed[i] = True
                station_load += times[i]
                placed_any = True
                for j in adj[i]:
                    remaining_in_deg[j] -= 1
        
        if not placed_any:
            num_stations += 1
            station_load = 0.0
            continue
        
        if all(completed):
            break
    
    return num_stations


def priority(instance):
    """
    Score each assembly task using bottleneck density:
    priority[i] = rank_down[i] * (1 + times[i] / min_stations_downstream[i])
    
    where rank_down is the longest path from task to any sink,
    and min_stations_downstream is the minimum number of stations needed
    to complete the downstream subtree of task i.
    
    This penalizes long tasks that sit on critical paths where station
    capacity is scarce.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]

    preds = instance["predecessors"]
    adj = [[] for _ in range(n)]
    in_deg = np.zeros(n, dtype=int)
    succ_count = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            adj[p].append(i)
            in_deg[i] += 1
            succ_count[p] += 1

    order = _topological_sort(adj, in_deg.copy(), n)

    # Downstream rank: longest path from task to any sink (including task's own time)
    rank_down = np.zeros(n, dtype=float)
    # Smooth (log-sum-exp) over successors to get a differentiable-ish surrogate of max
    for i in reversed(order):
        succs = adj[i]
        if succs:
            vals = rank_down[succs]
            m = float(vals.max())
            # soft-max with temperature; small temp ~ max, larger temp ~ log-sum-exp
            soft = m + float(np.log(np.sum(np.exp(vals - m))))
            # damp toward hard max to keep critical-path emphasis
            soft = 0.85 * m + 0.15 * soft
        else:
            soft = 0.0
        rank_down[i] = times[i] + soft

    # Refined downstream-station estimate: combine path lower bound with
    # total-subtree-work lower bound (both are valid lower bounds on stations).
    # total_work_down[i] = times[i] + sum over descendants of times
    total_work_down = np.zeros(n, dtype=float)
    for i in reversed(order):
        s = 0.0
        for j in adj[i]:
            s += total_work_down[j]
        total_work_down[i] = times[i] + s

    path_lb = rank_down / cycle_time
    work_lb = total_work_down / cycle_time
    # Use the larger of the two lower bounds (still a lower bound)
    min_stations_est = np.maximum(1.0, np.maximum(path_lb, work_lb))

    # Bottleneck density: ratio of task time to estimated stations for its subtree
    bottleneck_density = times / min_stations_est

    # Unlocking power: how many immediate successors a task unlocks.
    # Normalise by max to keep scale comparable; use a smooth log(1+x) to
    # avoid over-weighting very high fan-out.
    unlock = np.log1p(succ_count.astype(float))
    max_unlock = float(unlock.max()) if n else 1.0
    if max_unlock <= 0:
        max_unlock = 1.0
    unlock_norm = unlock / max_unlock

    # Combine: primary priority is downstream rank (critical path length),
    # boosted by bottleneck density and a mild unlocking term.
    # priority[i] = rank_down[i] * (1.0 + bottleneck_density[i]) + alpha * unlock_norm * rank_down[i]
    # The additive unlock term rewards tasks that open many branches, helping
    # the greedy filler keep more options available in later stations.
    alpha = 0.25
    priority_scores = rank_down * (1.0 + bottleneck_density) + alpha * unlock_norm * rank_down

    return priority_scores

# EVOLVE-BLOCK-END
