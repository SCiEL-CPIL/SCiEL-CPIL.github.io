"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import math


def priority(instance):
    """
    Compute priority scores using 'bottleneck density'.
    
    Priority = task_time / min_remaining_stations
    where min_remaining_stations is estimated by the critical path length downstream.
    
    This prioritizes tasks that have high processing time relative to the number of 
    stations they effectively occupy on the critical path.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    
    if n == 0:
        return np.array([])

    # Build adjacency lists
    preds = instance.get("predecessors", [None] * n)
    succ = [[] for _ in range(n)]
    in_degree = [0] * n
    
    for i in range(n):
        if preds[i] is not None:
            for p in preds[i]:
                succ[p].append(i)
                in_degree[i] += 1

    # Topological sort
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
                
    # If there's a cycle (shouldn't happen in valid ALB), handle gracefully
    if len(topo_order) != n:
        # Fallback to simple time priority if graph is invalid
        return times.copy()

    # Calculate Critical Path Length (downstream)
    # cp_len[i] = times[i] + max(cp_len[s] for s in succ[i])
    cp_len = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        if succ[node]:
            max_succ_cp = max(cp_len[s] for s in succ[node])
            cp_len[node] = times[node] + max_succ_cp
        else:
            cp_len[node] = times[node]
            
    # Calculate min remaining stations
    # The critical path must span at least ceil(cp_len / cycle_time) stations
    # We use this as a proxy for "scarce capacity" downstream
    min_stations = np.maximum(1.0, np.ceil(cp_len / cycle_time))
    
    # Bottleneck Density
    # High density = High time / Low station span = Acute bottleneck
    density = times / min_stations
    
    # Tie-breaker: Prefer tasks with more immediate successors (more branching/criticality)
    succ_counts = np.array([len(s) for s in succ], dtype=float)
    if succ_counts.max() > 0:
        # Normalize to [0, 1] and add small weight
        tie_breaker = 0.01 * (succ_counts / succ_counts.max())
    else:
        tie_breaker = np.zeros(n)
        
    priority_scores = density + tie_breaker
    
    return priority_scores
# EVOLVE-BLOCK-END
