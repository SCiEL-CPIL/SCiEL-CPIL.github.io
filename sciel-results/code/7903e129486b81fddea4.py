"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0

def priority(instance):
    """
    Computes priority scores for activities using a combination of
    Critical Path Method (CPM) Total Float and Resource Pressure.
    
    Higher score = Higher Priority.
    """
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        return np.asarray(instance["durations"], dtype=float)

    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]
    
    n = len(durations)
    
    # Build successor list for backward pass
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # 1. Forward Pass: Earliest Start (EST) and Earliest Finish (EFT)
    est = np.zeros(n)
    eft = np.zeros(n)
    
    # Topological order using Kahn's algorithm or simple DFS
    # Since N is small, we can do a simple iterative relaxation or DFS
    # Let's use a stack for DFS to get topological order
    visited = [False] * n
    topo_order = []
    
    def dfs(u):
        visited[u] = True
        for v in successors[u]:
            if not visited[v]:
                dfs(v)
        topo_order.append(u)
        
    for i in range(n):
        if not visited[i]:
            dfs(i)
            
    # Process in reverse topological order? No, forward pass needs topological order.
    # The DFS above produces nodes in reverse topological order (post-order).
    # So we reverse it to get topological order.
    topo_order.reverse()
    
    for u in topo_order:
        # EST is max of EFT of all predecessors
        max_pred_eft = 0.0
        for pred in predecessors[u]:
            if eft[pred] > max_pred_eft:
                max_pred_eft = eft[pred]
        est[u] = max_pred_eft
        eft[u] = est[u] + durations[u]
        
    # 2. Backward Pass: Latest Finish (LFT) and Latest Start (LST)
    # Project makespan is max EFT
    project_makespan = np.max(eft)
    
    lft = np.full(n, project_makespan)
    lst = np.full(n, project_makespan)
    
    # Process in reverse topological order (sinks to sources)
    for u in reversed(topo_order):
        # LFT is min of LST of all successors
        min_succ_lst = project_makespan
        for succ in successors[u]:
            if lst[succ] < min_succ_lst:
                min_succ_lst = lst[succ]
        lft[u] = min_succ_lst
        lst[u] = lft[u] - durations[u]
        
    # 3. Calculate Total Float (TF)
    # TF = LST - EST or LFT - EFT
    tf = lst - est
    
    # 4. Calculate Resource Pressure (RP)
    # Sum of (demand / capacity) for each resource
    # Handle zero capacities to avoid division by zero
    safe_capacities = np.where(capacities > 0, capacities, 1.0)
    # demands shape is (n, m), capacities shape is (m,)
    # We want sum over resources for each activity
    resource_pressure = np.sum(demands / safe_capacities, axis=1)
    
    # 5. Combine Scores
    # Priority = Inverse Float + Resource Pressure
    # Inverse Float: 1 / (TF + epsilon) to avoid division by zero and emphasize low TF
    # We add 1.0 to TF to keep values reasonable and positive
    inv_float = 1.0 / (tf + 1.0)
    
    # Normalize Resource Pressure? 
    # The magnitude of RP depends on capacities. 
    # If capacities are large, RP is small. If capacities are small, RP is large.
    # The term 1/(TF+1) is roughly in [0, 1].
    # We should scale RP to be comparable.
    # Let's check the scale of RP.
    # If RP is very large, it dominates. If very small, it's ignored.
    # A simple scaling: divide by max(RP) if max > 0
    max_rp = np.max(resource_pressure)
    if max_rp > 0:
        scaled_rp = resource_pressure / max_rp
    else:
        scaled_rp = np.zeros(n)
        
    # Weighting: 
    # Critical path (float) is generally more important for makespan.
    # Resource pressure helps with contention.
    # Let's try weight 1.0 for float and 0.5 for pressure.
    scores = inv_float + 0.5 * scaled_rp
    
    return scores

# EVOLVE-BLOCK-END
