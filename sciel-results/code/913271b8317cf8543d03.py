import numpy as np
import time

def priority(instance):
    start_time = time.time()
    deadline = 110.0  # Safety margin

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    predecessors = instance["predecessors"]
    demands = np.asarray(instance["demands"], dtype=float)
    if "capacities" in instance:
        capacities = np.asarray(instance["capacities"], dtype=float)
    else:
        # Fallback for potential typo in key name, though 'capacities' is standard
        capacities = np.asarray(instance.get("capancies", [1.0] * (demands.shape[1] if demands.ndim > 1 else 1)), dtype=float)

    # 1. Build successor lists and in-degrees
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1

    # 2. Topological Sort
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
        # Cycle detected or error, fallback
        return durations.copy()
    
    # 3. Forward Pass: Compute Earliest Start (ES) and Earliest Finish (EF)
    ES = np.zeros(n)
    EF = np.zeros(n)
    for i in topo_order:
        max_pred_ef = 0.0
        for p in predecessors[i]:
            if EF[p] > max_pred_ef:
                max_pred_ef = EF[p]
        ES[i] = max_pred_ef
        EF[i] = ES[i] + durations[i]
        
    project_duration = np.max(EF)
    
    # 4. Backward Pass: Compute Latest Finish (LF) and Latest Start (LS)
    LF = np.full(n, project_duration)
    LS = np.full(n, project_duration)
    
    for i in reversed(topo_order):
        if successors[i]:
            min_succ_ls = float('inf')
            for s in successors[i]:
                if LS[s] < min_succ_ls:
                    min_succ_ls = LS[s]
            LF[i] = min_succ_ls
        LS[i] = LF[i] - durations[i]
        
    # 5. Compute Resource Intensity
    safe_caps = np.where(capacities == 0, 1.0, capacities)
    # RI_i = sum(demand_ij / capacity_j)
    resource_intensity = np.sum(demands / safe_caps[np.newaxis, :], axis=1)
    
    # 6. Compute Transitive Successor Count
    # succ_count[i] = number of activities that depend on i (directly or indirectly)
    succ_count = np.zeros(n, dtype=int)
    # Process in reverse topological order to accumulate counts
    for i in reversed(topo_order):
        for s in successors[i]:
            succ_count[i] += 1 + succ_count[s]
            
    # 7. Normalize Metrics
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        range_val = max_val - min_val
        if range_val < 1e-10:
            return np.zeros_like(arr)
        return (arr - min_val) / range_val

    # Urgency: Higher is better (smaller LF)
    urgency = project_duration - LF
    urgency_norm = normalize(urgency)
    
    # Resource Intensity: Higher is better (more resource constrained)
    ri_norm = normalize(resource_intensity)
    
    # Successor Count: Higher is better (unblocks more activities)
    sc_norm = normalize(succ_count.astype(float))
    
    # Duration: Shorter is better (SPT tiebreaker)
    dur_norm = normalize(-durations) # Invert so shorter is higher
    
    # 8. Combine Scores
    # Weights: Urgency is most important, then Resource Load, then Successor Unblocking, then Duration
    scores = 0.55 * urgency_norm + 0.25 * ri_norm + 0.15 * sc_norm + 0.05 * dur_norm
    
    return scores
