import numpy as np
import time

def priority(instance):
    start_time = time.time()
    deadline = 120.0  # Safety margin
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    predecessors = instance["predecessors"]
    
    # 1. Build successor lists
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1
            
    # 2. Topological Sort (Kahn's Algorithm)
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    current_queue = queue[:]
    in_degree_temp = in_degree[:]
    
    while current_queue:
        # Pick a node (order doesn't matter for correctness of L_start/L_end calculation, 
        # but for determinism, we can sort or just pop)
        node = current_queue.pop(0) 
        topo_order.append(node)
        for succ in successors[node]:
            in_degree_temp[succ] -= 1
            if in_degree_temp[succ] == 0:
                current_queue.append(succ)
                
    if len(topo_order) != n:
        # Cycle detected? Fallback to durations
        return durations.copy()

    # 3. Compute L_start (Longest path from start to i)
    L_start = np.zeros(n)
    for i in topo_order:
        # L_start[i] is duration[i] + max(L_start[p]) for p in predecessors[i]
        max_pred = 0.0
        for p in predecessors[i]:
            if L_start[p] > max_pred:
                max_pred = L_start[p]
        L_start[i] = durations[i] + max_pred
        
    # 4. Compute L_end (Longest path from i to end)
    L_end = np.zeros(n)
    # Iterate in reverse topological order
    for i in reversed(topo_order):
        # L_end[i] is duration[i] + max(L_end[s]) for s in successors[i]
        max_succ = 0.0
        for s in successors[i]:
            if L_end[s] > max_succ:
                max_succ = L_end[s]
        L_end[i] = durations[i] + max_succ
        
    # 5. Calculate Priority Score using multiple heuristics
    # Criticality = L_start + L_end - duration
    criticality = L_start + L_end - durations
    
    # Resource factor: dynamic capacity depletion / bottleneck congestion model
    demands = np.asarray(instance["demands"], dtype=float)  # shape (n, num_resources)
    capacities = np.asarray(instance["capacities"], dtype=float)  # shape (num_resources,)
    num_resources = len(capacities)
    
    # For each resource, compute the total project-wide demand as a proxy for
    # how "tight" that resource is. Resources with higher total demand relative
    # to capacity are more likely to be bottlenecks.
    total_demand_per_resource = np.sum(demands, axis=0)  # shape (num_resources,)
    resource_tightness = total_demand_per_resource / (capacities + 1e-10)  # shape (num_resources,)
    
    # For each activity, its congestion score is the max over resources of
    # (demand[i,r] / capacity[r]) * resource_tightness[r].
    # This captures: an activity using a resource that is globally scarce
    # (high tightness) AND has high individual demand on that resource
    # creates the most congestion.
    # Normalize demands by capacity per resource
    normalized_demands = demands / (capacities[np.newaxis, :] + 1e-10)  # shape (n, num_resources)
    
    # Weight each resource's contribution by its global tightness
    weighted_congestion = normalized_demands * resource_tightness[np.newaxis, :]  # shape (n, num_resources)
    
    # Bottleneck congestion: max over resources (fluid-flow peak)
    # Use a soft-max via log-sum-exp for smoothness, then normalize
    # Actually, max is appropriate here for "peak congestion" - use max
    resource_congestion = np.max(weighted_congestion, axis=1)  # shape (n,)
    
    # Normalize congestion to [0, 1] range for stable weighting
    max_congestion = np.max(resource_congestion) if np.max(resource_congestion) > 0 else 1.0
    resource_congestion_norm = resource_congestion / (max_congestion + 1e-10)
    
    # Duration factor: longer activities should be prioritized
    duration_norm = durations / (np.max(durations) + 1e-10)
    
    # Successor count: activities with more successors unblock more work
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    max_succ = np.max(succ_count) if np.max(succ_count) > 0 else 1.0
    succ_norm = succ_count / (max_succ + 1e-10)
    
    # Combine heuristics with weighted sum
    # Criticality dominates; bottleneck congestion is second; duration and successor count are tiebreakers
    scores = 1.0 * criticality + 0.5 * resource_congestion_norm + 0.15 * duration_norm + 0.1 * succ_norm
    
    return scores
