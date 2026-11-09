"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(instance):
    """
    Computes priority scores for activities using a hybrid heuristic:
    Longest Duration / (Total Slack + 1).
    
    This prioritizes activities that are both long and have little scheduling flexibility (low slack),
    which is a known effective strategy for minimizing makespan in Resource-Constrained Project Scheduling Problems (RCPSP).
    """
    # Deadline guard (though computation is O(N+E) and very fast)
    start_time = time.time()
    
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    
    n = len(durations)
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            
    # Forward pass: Calculate ES and EF
    # ES[i] = max(EF[p] for p in predecessors[i])
    # EF[i] = ES[i] + D[i]
    ES = np.zeros(n)
    EF = np.zeros(n)
    
    # We need to process in topological order.
    # Since we don't have an explicit topological order, we can do a simple iterative pass
    # or use Kahn's algorithm. For N=30, a simple loop until stable is fine, 
    # but let's do a proper topological sort to be safe and fast.
    
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        in_degree[i] = len(predecessors[i])
        
    queue = []
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
            
    topo_order = []
    while queue:
        # Pick a node (order doesn't strictly matter for correctness of ES/EF if we update neighbors)
        # But for efficiency, we can just pop
        u = queue.pop(0) 
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # If there's a cycle, topo_order will be incomplete, but RCPSP is a DAG.
    # Calculate ES/EF in topological order
    for u in topo_order:
        if predecessors[u]:
            ES[u] = max(EF[p] for p in predecessors[u])
        else:
            ES[u] = 0.0
        EF[u] = ES[u] + durations[u]
        
    # Projected makespan (ignoring resources)
    C_max = np.max(EF) if n > 0 else 0.0
    
    # Backward pass: Calculate LS and LF
    # LF[i] = min(LS[s] for s in successors[i])
    # LS[i] = LF[i] - D[i]
    LS = np.zeros(n)
    LF = np.zeros(n)
    
    # Process in reverse topological order
    for u in reversed(topo_order):
        if successors[u]:
            LF[u] = min(LS[s] for s in successors[u])
        else:
            LF[u] = C_max
        LS[u] = LF[u] - durations[u]
        
    # Calculate Total Slack
    Slack = LS - ES
    
    # Priority Score: Duration / (Slack + 1)
    # This boosts long activities that are critical (low slack)
    # Adding 1 to slack avoids division by zero and ensures positive scores
    scores = durations / (Slack + 1.0)
    
    # Check time (should be well under 120s)
    elapsed = time.time() - start_time
    if elapsed > 110:
         # Fallback to simple duration if something went wrong (unlikely)
         return durations
         
    return scores
# EVOLVE-BLOCK-END
