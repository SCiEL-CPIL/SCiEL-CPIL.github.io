import numpy as np

def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]
    
    n = len(durations)
    
    # 1. Build successors list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            
    # 2. Compute Longest Path to End (Critical Path Length)
    path_to_end = [0.0] * n
    visited = [False] * n
    in_stack = [False] * n
    
    def dfs(u):
        if visited[u]:
            return path_to_end[u]
        if in_stack[u]:
            return 0.0
        in_stack[u] = True
        max_succ_path = 0.0
        for v in successors[u]:
            v_path = dfs(v)
            if v_path > max_succ_path:
                max_succ_path = v_path
        path_to_end[u] = durations[u] + max_succ_path
        in_stack[u] = False
        visited[u] = True
        return path_to_end[u]
        
    for i in range(n):
        dfs(i)
        
    # 3. Compute Resource Pressure
    safe_capacities = np.where(capacities == 0, 1.0, capacities)
    pressure_matrix = demands / safe_capacities
    pressure = np.sum(pressure_matrix, axis=1)
    
    # 4. Combine Scores with Normalization
    # Normalize to ensure both components contribute significantly
    max_cpl = max(path_to_end) if max(path_to_end) > 0 else 1.0
    max_press = max(pressure) if max(pressure) > 0 else 1.0
    
    norm_cpl = np.asarray(path_to_end) / max_cpl
    norm_press = pressure / max_press
    
    # Weighting: Critical path is primary, but resource pressure is weighted
    # to identify bottlenecks.
    scores = norm_cpl + 1.5 * norm_press
    
    return scores
