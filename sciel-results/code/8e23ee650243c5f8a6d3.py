import numpy as np
from collections import deque

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    # Build adjacency list for successors
    # preds[i] is a list of predecessors for task i
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Topological sort (Kahn's algorithm)
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
            
    topo_order = []
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
                
    # If topological sort is not complete, there's a cycle. 
    # Assuming valid input, this shouldn't happen. 
    # If it does, we fall back to a simple heuristic.
    if len(topo_order) < n:
        # Fallback: LTF
        return times
        
    # Compute number of transitive successors
    # We process in reverse topological order
    # succ_count[i] = number of tasks that depend on i (transitively)
    succ_count = [0] * n
    
    # Reverse topological order
    for u in reversed(topo_order):
        # succ_count[u] is the sum of succ_count[v] for all immediate successors v
        # Plus 1 for each immediate successor? No, transitive means all downstream.
        # Let's define S(u) = set of all successors of u (transitive).
        # |S(u)| = sum(|S(v)|) for v in successors(u) is incorrect if there are shared successors.
        # We need to count unique successors.
        
        # Alternative: Compute "Rank" or "Level"
        # Level(u) = times[u] + max(Level(v) for v in successors(u))
        # This is the longest path from u to end.
        # Prioritizing by Level is a standard heuristic (Critical Path Method).
        
        # Let's try Level (Longest Path) as the primary metric.
        # It captures both duration and depth.
        
        pass

    # Let's compute Level (Longest Path to End)
    # Level(u) = times[u] + max(Level(v) for v in successors(u))
    # Base case: if no successors, Level(u) = times[u]
    
    levels = [0.0] * n
    for u in reversed(topo_order):
        max_succ_level = 0.0
        for v in successors[u]:
            if levels[v] > max_succ_level:
                max_succ_level = levels[v]
        levels[u] = times[u] + max_succ_level
        
    # Now, let's also compute the number of immediate successors
    imm_succ_count = [len(succ) for succ in successors]
    
    # Combine: Level is a good measure of "criticality".
    # But "Most Successors" was found to be effective.
    # Let's try a combination: 
    # Priority = Level * 1000 + imm_succ_count
    # Or just Level?
    
    # The knowledge said "transitive successors" was good.
    # Level is correlated with the path length, not the breadth.
    # A task with many immediate successors but short times might have a lower Level than a task with few successors but very long times.
    # However, "Most Successors" (breadth) helps with unlocking.
    
    # Let's try to compute the number of transitive successors properly.
    # Since N=30, we can use bitmasks or sets.
    # Bitmasks are efficient for N<=60.
    
    succ_mask = [0] * n
    # Process in reverse topological order
    for u in reversed(topo_order):
        mask = 0
        for v in successors[u]:
            mask |= succ_mask[v]
            mask |= (1 << v) # Include immediate successor
        succ_mask[u] = mask
        
    trans_succ_count = [bin(mask).count('1') for mask in succ_mask]
    
    # Priority: 
    # 1. Transitive Successors Count (Breadth)
    # 2. Level (Depth/Criticality)
    # 3. Duration
    
    # Let's weight them.
    # High breadth is good for unlocking.
    # High level is good for critical path.
    
    # Let's try: priority = trans_succ_count * 1000 + levels
    # This prioritizes breadth first, then depth.
    
    priorities = np.zeros(n)
    for i in range(n):
        priorities[i] = trans_succ_count[i] * 1000.0 + levels[i]
        
    return priorities
