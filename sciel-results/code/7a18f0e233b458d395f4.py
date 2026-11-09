import numpy as np
from collections import defaultdict, deque

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    # Build successors graph
    successors = defaultdict(list)
    in_degree = [0] * n
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Topological Sort (Kahn's algorithm)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    temp_in_degree = in_degree[:]
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            temp_in_degree[v] -= 1
            if temp_in_degree[v] == 0:
                queue.append(v)
                
    # If graph has cycle, topo_order might be incomplete. 
    # But ALB graphs are DAGs.
    # Reverse topological order for RPW calculation
    rpw = np.zeros(n)
    for u in reversed(topo_order):
        # rpw[u] = times[u] + sum(rpw[v] for v in successors[u])
        s = 0.0
        for v in successors[u]:
            s += rpw[v]
        rpw[u] = times[u] + s
        
    # Calculate immediate successor count
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Calculate total descendant count (optional, might help)
    # For small N, we can do DFS
    # But let's stick to immediate successors first to match the "inspiration"
    
    # Score: 
    # Dominant: RPW
    # Secondary: Task Time (tie breaker / packing)
    # Tertiary: Successor Count (branching factor)
    
    # Weights
    w_rpw = 1.0
    w_time = 0.1
    w_succ = 0.01
    
    scores = w_rpw * rpw + w_time * times + w_succ * succ_count
    
    return scores
