import numpy as np
from collections import deque

def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(durations)
    
    # Handle case where n is 0
    if n == 0:
        return np.array([])
    
    # Extract resource info
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    n_resources = len(capacities) if capacities is not None else 0
    
    # Build successors list and in-degrees
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Topological Sort (Kahn's Algorithm)
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
            
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # If there's a cycle, topo_order will be incomplete. 
    # RCPSP instances are DAGs, so this shouldn't happen.
    # But just in case, if len(topo_order) < n, we might have issues.
    # For safety, if incomplete, we fall back to LDF?
    if len(topo_order) < n:
         return durations

    # Calculate Longest Path to End (LPE / LD)
    LPE = np.zeros(n)
    # Process in reverse topological order
    for u in reversed(topo_order):
        max_succ_LPE = 0.0
        for v in successors[u]:
            if LPE[v] > max_succ_LPE:
                max_succ_LPE = LPE[v]
        LPE[u] = durations[u] + max_succ_LPE

    # Calculate Longest Path from Start (LPS)
    LPS = np.zeros(n)
    # Process in forward topological order
    for u in topo_order:
        max_pred_LPS = 0.0
        for p in predecessors[u]:
            if LPS[p] > max_pred_LPS:
                max_pred_LPS = LPS[p]
        LPS[u] = max_pred_LPS + durations[u]

    # Calculate demand-to-capacity ratio (resource pressure)
    # Higher ratio means more likely to cause resource contention
    resource_pressure = np.zeros(n)
    if demands is not None and capacities is not None and n_resources > 0:
        demands_arr = np.asarray(demands, dtype=float)
        capacities_arr = np.asarray(capacities, dtype=float)
        # For each activity, compute max demand/capacity ratio across resources
        # This identifies activities that are most likely to bottleneck resources
        for i in range(n):
            if i < len(demands_arr):
                for r in range(n_resources):
                    if capacities_arr[r] > 0:
                        ratio = demands_arr[i][r] / capacities_arr[r]
                        if ratio > resource_pressure[i]:
                            resource_pressure[i] = ratio

    # Combine LPS and LPE into a Path Score
    # This metric prioritizes activities on the longest paths through the project
    path_score = LPS + LPE

    # Normalize path score and resource pressure
    ps_min, ps_max = path_score.min(), path_score.max()
    ps_range = ps_max - ps_min if ps_max > ps_min else 1.0
    ps_norm = (path_score - ps_min) / ps_range

    rp_min, rp_max = resource_pressure.min(), resource_pressure.max()
    rp_range = rp_max - rp_min if rp_max > rp_min else 1.0
    rp_norm = (resource_pressure - rp_min) / rp_range

    # Weighted combination: 80% path criticality, 20% resource pressure
    # This ensures activities on critical paths are prioritized, with a boost
    # for those that also consume significant resources
    combined_score = 0.8 * ps_norm + 0.2 * rp_norm

    # Add tiny tie-breaker for numerical stability
    return combined_score + durations * 1e-9
