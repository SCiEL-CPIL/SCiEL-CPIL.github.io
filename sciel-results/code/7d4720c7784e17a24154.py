"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    """
    Compute priority scores using a 'bottleneck density' heuristic.
    
    For each task, we compute:
    1. The critical path length downstream (longest path from this task to any leaf)
    2. An estimate of the minimum number of stations needed to complete the downstream subtree
    
    The priority is defined as:
        priority[i] = times[i] / min_stations_downstream[i]
    
    This penalizes long tasks that sit on critical paths where station capacity is scarce.
    We add a small tie-breaker based on the number of immediate successors.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    
    # Build successor list: succ[i] = list of tasks that depend on i
    preds = instance.get("predecessors", [None] * n)
    succ = [[] for _ in range(n)]
    for i in range(n):
        if preds[i] is not None:
            for p in preds[i]:
                succ[p].append(i)
    
    # Compute in-degrees for topological sort
    in_degree = [0] * n
    for i in range(n):
        if preds[i] is not None:
            for p in preds[i]:
                in_degree[i] += 1
    
    # Topological sort using Kahn's algorithm
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    temp_in_degree = in_degree[:]
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in succ[node]:
            temp_in_degree[s] -= 1
            if temp_in_degree[s] == 0:
                queue.append(s)
    
    # Compute critical path downstream: longest path from node to any leaf
    # crit_down[node] = times[node] + max(crit_down[s] for s in succ[node]) if succ[node] else times[node]
    crit_down = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        if succ[node]:
            crit_down[node] = times[node] + max(crit_down[s] for s in succ[node])
        else:
            crit_down[node] = times[node]
    
    # Compute total work in downstream subtree (including the node itself)
    subtree_work = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        subtree_work[node] = times[node] + sum(subtree_work[s] for s in succ[node])
    
    # Estimate minimum number of stations needed for downstream subtree
    # Use the maximum of:
    #   - ceil(subtree_work / cycle_time)  (total work constraint)
    #   - ceil(crit_down / cycle_time)     (critical path constraint)
    # This gives a more accurate lower bound on required stations
    work_stations = np.ceil(subtree_work / cycle_time)
    path_stations = np.ceil(crit_down / cycle_time)
    min_stations = np.maximum(work_stations, path_stations)
    min_stations = np.maximum(min_stations, 1.0)  # At least 1 station
    
    # Bottleneck density: ratio of task time to min remaining stations
    # Higher values indicate tasks that are long relative to available station capacity
    bottleneck_density = times / min_stations
    
    # Add a small tie-breaker based on the number of immediate successors
    # Tasks with more successors are more critical and should be scheduled earlier
    succ_counts = np.array([len(s) for s in succ], dtype=float)
    max_succ = succ_counts.max() if n > 0 else 1.0
    if max_succ > 0:
        succ_bonus = succ_counts / max_succ
    else:
        succ_bonus = np.zeros(n)
    
    # Final priority: bottleneck density + small weight on successor count
    priority_scores = bottleneck_density + 0.05 * succ_bonus
    
    return priority_scores
# EVOLVE-BLOCK-END
