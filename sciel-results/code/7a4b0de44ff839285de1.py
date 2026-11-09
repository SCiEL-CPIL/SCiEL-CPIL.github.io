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
    # We need topological order. Since predecessors are given, 
    # we can use in-degrees to find topological sort, or just do a DFS/memoization.
    # Given n=30, simple recursion with memoization is fine.
    
    path_to_end = [0.0] * n
    visited = [False] * n
    in_stack = [False] * n
    
    def dfs(u):
        if visited[u]:
            return path_to_end[u]
        if in_stack[u]:
            # Cycle detected? RCPSP usually assumes DAG.
            # If cycle, return 0 to avoid infinite loop.
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
    # demands shape: (n, num_resources)
    # capacities shape: (num_resources,)
    # pressure[i] = sum(demands[i][r] / capacities[r])
    
    # Avoid division by zero
    safe_capacities = np.where(capacities == 0, 1.0, capacities)
    # pressure is sum of ratios per resource for each activity
    # demands[i, r] / capacities[r]
    # Result shape (n, num_resources)
    pressure_matrix = demands / safe_capacities
    # Sum over resources (axis=1)
    pressure = np.sum(pressure_matrix, axis=1)
    
    # 4. Combine Scores
    # We want high priority for high path_to_end and high pressure.
    # Score = path_to_end + pressure
    # Scaling pressure might be needed if durations are much larger than pressure ratios.
    # Typical durations: 1-10. Path length: up to 300. Pressure: 0-5.
    # Path length dominates. This is generally good (Critical Path First).
    # To give more weight to resources, we could multiply pressure by a factor.
    # Let's use a small multiplier to break ties or slightly boost resource-heavy tasks.
    # Or just add them.
    
    scores = np.asarray(path_to_end) + pressure
    
    return scores
