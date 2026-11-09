# EVOLVE-BLOCK-START
import numpy as np
from collections import deque

def priority(instance):
    """
    Score each activity for a serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Approach: Minimum Total Slack (MTS) combined with Resource Demand Intensity.
    1. Compute Early Start/Finish (Forward Pass).
    2. Compute Late Start/Finish (Backward Pass).
    3. Calculate Total Slack.
    4. Combine Inverse Total Slack and Resource Demand.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    preds = instance.get("predecessors", None)
    
    # Build successor map
    succs = [[] for _ in range(n)]
    in_deg = [0] * n
    if preds is not None:
        for j, pred_list in enumerate(preds):
            in_deg[j] = len(pred_list)
            for i in pred_list:
                succs[i].append(j)
    
    # Topological sort (Kahn's algorithm)
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    in_deg_copy = in_deg[:] # Keep original for later if needed, though we only need it once
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in succs[node]:
            in_deg_copy[s] -= 1
            if in_deg_copy[s] == 0:
                queue.append(s)
    
    if len(topo_order) < n:
        topo_order = list(range(n))
    
    # 1. Forward Pass: Early Start (ES) and Early Finish (EF)
    ES = np.zeros(n, dtype=float)
    EF = np.zeros(n, dtype=float)
    for node in topo_order:
        if not preds or not preds[node]:
            ES[node] = 0.0
        else:
            # ES is max of EF of predecessors
            max_ef = 0.0
            for p in preds[node]:
                if EF[p] > max_ef:
                    max_ef = EF[p]
            ES[node] = max_ef
        EF[node] = ES[node] + durations[node]
    
    # Project Duration T
    T = np.max(EF) if n > 0 else 0.0
    
    # 2. Backward Pass: Late Finish (LF) and Late Start (LS)
    LF = np.zeros(n, dtype=float)
    LS = np.zeros(n, dtype=float)
    # Initialize LF to T for sinks
    for node in range(n):
        if not succs[node]:
            LF[node] = T
        else:
            LF[node] = T # Initialize to T, will be updated to min of successors' LS
    
    # Process in reverse topological order
    for node in reversed(topo_order):
        if succs[node]:
            min_ls = float('inf')
            for s in succs[node]:
                # LS[s] must be calculated already because we are going reverse topo
                if LS[s] < min_ls:
                    min_ls = LS[s]
            LF[node] = min_ls
        LS[node] = LF[node] - durations[node]
        
        # Ensure LS is not negative (though with valid DAGs and T=max EF, it shouldn't be for critical path, 
        # but for non-critical it might be if we made a mistake. Actually LS >= 0 always if T is correct max EF).
        if LS[node] < 0:
            LS[node] = 0.0

    # 3. Total Slack (TS)
    TS = LS - ES
    
    # 4. Resource Demand Intensity
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    demand_score = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands, dtype=float)
        caps_arr = np.asarray(capacities, dtype=float)
        
        if demands_arr.ndim == 2:
            if len(caps_arr) == demands_arr.shape[1]:
                # Normalize demand by capacity
                # If capacity is 0, treat demand as 0 or high? Usually capacity > 0.
                safe_caps = np.where(caps_arr > 0, caps_arr, 1.0)
                norm_demands = demands_arr / safe_caps
                # Max normalized demand across resources for this activity
                demand_score = np.max(norm_demands, axis=1)
            else:
                demand_score = np.max(demands_arr, axis=1)
        elif demands_arr.ndim == 1:
            if caps_arr.size > 0:
                cap = caps_arr.flat[0]
                if cap > 0:
                    demand_score = demands_arr / cap
                else:
                    demand_score = demands_arr
            else:
                demand_score = demands_arr
    
    # Normalize components
    def normalize(x):
        x = np.asarray(x, dtype=float)
        x_min = np.min(x)
        x_max = np.max(x)
        if x_max - x_min < 1e-9:
            return np.ones_like(x) * 0.5
        return (x - x_min) / (x_max - x_min)
    
    # Inverse Total Slack: Lower TS is better (higher priority).
    # We want to maximize priority, so we invert TS.
    # Using (T - TS) is equivalent to prioritizing activities closer to the critical path in terms of slack.
    # Or simply -TS. Let's use a normalized negative slack.
    # Since TS >= 0, lower TS is better.
    # Priority component 1: -TS (normalized)
    ts_norm = normalize(TS)
    slack_priority = 1.0 - ts_norm # Higher when TS is low
    
    # Priority component 2: Demand
    dem_norm = normalize(demand_score)
    
    # Weighted combination
    # MTS is generally a strong heuristic. Resource demand helps break ties and handle resource contention.
    # Let's try 0.7 for Slack and 0.3 for Demand.
    score = 0.7 * slack_priority + 0.3 * dem_norm
    
    return score
# EVOLVE-BLOCK-END
