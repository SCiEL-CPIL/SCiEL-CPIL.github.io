import numpy as np
import time

def priority(instance):
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
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
        max_pred = 0.0
        for p in predecessors[i]:
            if L_start[p] > max_pred:
                max_pred = L_start[p]
        L_start[i] = durations[i] + max_pred
        
    # 4. Compute L_end (Longest path from i to end)
    L_end = np.zeros(n)
    for i in reversed(topo_order):
        max_succ = 0.0
        for s in successors[i]:
            if L_end[s] > max_succ:
                max_succ = L_end[s]
        L_end[i] = durations[i] + max_succ
    
    # 4b. Compute slack-based priority: activities with less slack are more critical
    # Slack = L_end - L_start (time window flexibility)
    slack = L_end - L_start
    
    # Inverse slack: lower slack = higher priority
    # Use a smooth approximation to avoid division by zero
    min_slack = np.min(slack)
    max_slack = np.max(slack)
    slack_range = max_slack - min_slack + 1e-10
    
    # Normalize slack to [0, 1], then invert
    slack_norm = (slack - min_slack) / slack_range
    inv_slack = 1.0 - slack_norm
        
    # 5. Calculate Priority Score using multiple heuristics
    # Criticality = L_start + L_end - duration
    criticality = L_start + L_end - durations
    
    # Resource factor: prioritize activities with higher resource demand
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    
    # Normalize resource usage: sum of (demand/capacity) for each activity
    resource_usage = np.sum(demands / capacities[np.newaxis, :], axis=1)
    
    # Duration factor: longer activities should be prioritized
    duration_norm = durations / (np.max(durations) + 1e-10)
    
    # Successor count factor: prioritize activities that unlock many successors
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    succ_count_norm = succ_count / (np.max(succ_count) + 1e-10)
    
    # Predecessor count factor: prioritize activities with fewer predecessors (more flexible)
    pred_count = np.array([len(predecessors[i]) for i in range(n)], dtype=float)
    pred_count_norm = pred_count / (np.max(pred_count) + 1e-10)
    
    # Combine heuristics with weighted sum
    # Criticality is most important, then resource contention, then duration
    # Add small weight for successor count, inverse predecessor count, and inverse slack
    scores = 1.0 * criticality + 0.3 * resource_usage + 0.2 * duration_norm + 0.1 * succ_count_norm - 0.05 * pred_count_norm + 0.15 * inv_slack
    
    return scores
