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

    # Dynamic resource-contention priority:
    # For each activity, compute the maximum demand/capacity ratio across resources,
    # then identify which activities share resources with others (contention groups).
    # Activities in tight resource groups get boosted to schedule early.
    
    resource_pressure = np.zeros(n)
    contention_score = np.zeros(n)
    
    if demands is not None and capacities is not None and n_resources > 0:
        demands_arr = np.asarray(demands, dtype=float)
        capacities_arr = np.asarray(capacities, dtype=float)
        
        # Compute per-activity max demand/capacity ratio
        for i in range(n):
            if i < len(demands_arr):
                for r in range(n_resources):
                    if capacities_arr[r] > 0:
                        ratio = demands_arr[i][r] / capacities_arr[r]
                        if ratio > resource_pressure[i]:
                            resource_pressure[i] = ratio
        
        # Compute contention: for each activity, sum the ratios of other activities
        # that share at least one resource with it. This captures how much
        # resource pressure this activity faces from its peers.
        # Use a soft-max-like approach: weighted by shared resource count
        for i in range(n):
            if i >= len(demands_arr):
                continue
            for j in range(n):
                if i == j or j >= len(demands_arr):
                    continue
                # Check if i and j share any resource with non-zero demand
                shares = False
                for r in range(n_resources):
                    if demands_arr[i][r] > 0 and demands_arr[j][r] > 0:
                        shares = True
                        break
                if shares:
                    # Weight by the other activity's resource pressure
                    contention_score[i] += resource_pressure[j]
    
    # Normalize LPE and contention score
    lpe_min, lpe_max = LPE.min(), LPE.max()
    lpe_range = lpe_max - lpe_min if lpe_max > lpe_min else 1.0
    lpe_norm = (LPE - lpe_min) / lpe_range

    cs_min, cs_max = contention_score.min(), contention_score.max()
    cs_range = cs_max - cs_min if cs_max > cs_min else 1.0
    cs_norm = (contention_score - cs_min) / cs_range

    # Normalize resource pressure
    rp_min, rp_max = resource_pressure.min(), resource_pressure.max()
    rp_range = rp_max - rp_min if rp_max > rp_min else 1.0
    rp_norm = (resource_pressure - rp_min) / rp_range

    # Weighted combination:
    # - 50% LPE (critical path to end - schedule late-finishing tasks early)
    # - 30% contention (activities facing high peer resource pressure)
    # - 20% own resource pressure (activities that are themselves heavy)
    combined_score = 0.5 * lpe_norm + 0.3 * cs_norm + 0.2 * rp_norm

    # Add tiny tie-breaker for numerical stability
    return combined_score + durations * 1e-9
