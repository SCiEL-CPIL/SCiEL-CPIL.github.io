# EVOLVE-BLOCK-START
import numpy as np
from collections import deque

def priority(instance):
    """
    Score each activity for a serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Approach: 
    1. Compute Critical Path Method (CPM) parameters (ES, EF, LS, LF, Total Float).
    2. Prioritize activities with low Total Float (critical path).
    3. Break ties using Resource Intensity (high demand relative to capacity).
    """
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Parse predecessors
    preds = instance.get("predecessors", None)
    if preds is None:
        preds = [[] for _ in range(n)]
        
    # Build successor map
    succs = [[] for _ in range(n)]
    in_deg = [0] * n
    for j in range(n):
        pred_list = preds[j]
        in_deg[j] = len(pred_list)
        for i in pred_list:
            succs[i].append(j)
            
    # Topological Sort
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in succs[node]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)
                
    if len(topo_order) < n:
        # Cycle detected, fallback to index order
        topo_order = list(range(n))
        
    # Forward Pass: Early Start (ES), Early Finish (EF)
    ES = np.zeros(n, dtype=float)
    EF = np.zeros(n, dtype=float)
    for node in topo_order:
        if not preds[node]:
            ES[node] = 0.0
        else:
            max_ef = 0.0
            for p in preds[node]:
                if EF[p] > max_ef:
                    max_ef = EF[p]
            ES[node] = max_ef
        EF[node] = ES[node] + durations[node]
        
    # Project makespan (ignoring resources)
    makespan = max(EF) if n > 0 else 0.0
    
    # Backward Pass: Late Finish (LF), Late Start (LS)
    LF = np.zeros(n, dtype=float)
    LS = np.zeros(n, dtype=float)
    out_deg = [len(succs[i]) for i in range(n)]
    
    for node in reversed(topo_order):
        if out_deg[node] == 0:
            LF[node] = makespan
        else:
            min_ls_succ = makespan
            for s in succs[node]:
                if LS[s] < min_ls_succ:
                    min_ls_succ = LS[s]
            LF[node] = min_ls_succ
        LS[node] = LF[node] - durations[node]
        
    # Total Float
    Float = LS - ES
    
    # Resource Intensity
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    intensity = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands, dtype=float)
        caps_arr = np.asarray(capacities, dtype=float)
        
        if demands_arr.ndim == 1:
            if caps_arr.size == 1:
                cap = caps_arr.flat[0]
                if cap > 0:
                    intensity = demands_arr / cap
            else:
                intensity = demands_arr
        elif demands_arr.ndim == 2:
            # demands_arr[i, k]
            if len(caps_arr) == demands_arr.shape[1]:
                # Avoid division by zero
                safe_caps = np.where(caps_arr > 0, caps_arr, 1.0)
                norm_demands = demands_arr / safe_caps
                # Max normalized demand per activity
                intensity = np.max(norm_demands, axis=1)
            else:
                intensity = np.max(demands_arr, axis=1)
                
    # Normalize Intensity to [0, 1]
    if intensity.max() > intensity.min():
        intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    else:
        intensity_norm = np.zeros(n)
        
    # Score Calculation
    # We want higher score for lower Float.
    # Max Float helps in scaling.
    max_float = Float.max() if n > 0 else 0.0
    # Score = (Max_Float - Float) + small_weight * Intensity
    # This ensures critical tasks (Float=0) get high score (Max_Float).
    # Non-critical tasks get lower scores.
    # Intensity breaks ties.
    
    score = (max_float - Float) + 0.1 * intensity_norm
    
    return score
# EVOLVE-BLOCK-END
