import numpy as np
from collections import deque

def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]
    
    n = len(durations)
    if n == 0:
        return np.array([])
        
    R = len(capacities)
    
    # 1. Build adjacency lists
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    # 2. Topological Sort (Kahn's Algorithm)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    top_order = []
    while queue:
        u = queue.popleft()
        top_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # 3. CPM Forward Pass (Earliest Start/Finish)
    ES = [0.0] * n
    EF = [0.0] * n
    for u in top_order:
        for v in successors[u]:
            if EF[u] > ES[v]:
                ES[v] = EF[u]
        EF[u] = ES[u] + durations[u]
        
    project_duration = max(EF) if n > 0 else 0.0
    
    # 4. CPM Backward Pass (Latest Start/Finish)
    LS = [0.0] * n
    LF = [0.0] * n
    # Initialize sinks
    for u in top_order:
        if not successors[u]:
            LF[u] = project_duration
        else:
            LF[u] = float('inf')
            
    # Process in reverse topological order
    for u in reversed(top_order):
        if successors[u]:
            LF[u] = min(LF[v] for v in successors[u])
        LS[u] = LF[u] - durations[u]
        
    # 5. Calculate Slack
    Slack = np.array([LS[i] - ES[i] for i in range(n)])
    
    # 6. Calculate Resource Pressure
    safe_capacities = np.where(capacities == 0, 1.0, capacities)
    # Handle case where demands might be 1D? Assume 2D (n, R) based on previous code.
    # If demands is 1D, it implies 1 resource? 
    # Previous code: demands / safe_capacities. 
    # If demands is (n,) and capacities is (R,), this fails unless R=1 and broadcasting works or shapes match.
    # Assuming standard RCPSP format: demands[i][r].
    pressure = np.sum(demands / safe_capacities, axis=1)
    
    # 7. Combine Scores
    # High Priority = Low Slack (Critical) AND High Pressure (Bottleneck)
    # Score = (1 / (Slack + 1)) * (1 + Pressure)
    # This gives high weight to zero-slack tasks, boosted by resource pressure.
    
    scores = (1.0 / (Slack + 1.0)) * (1.0 + pressure)
    
    return scores
