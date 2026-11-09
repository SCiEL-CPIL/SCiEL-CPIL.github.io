# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    """
    Score each activity for a serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Approach: Total Float (Slack) Minimization combined with Resource Intensity.
    Activities with less slack (closer to the critical path) are prioritized.
    Among activities with similar slack, those with higher resource demand are prioritized
    to resolve resource conflicts earlier.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Get predecessors/successors
    preds = instance.get("predecessors", None)
    
    # Build successor map and in-degrees
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
    # We need a copy of in_deg for the second pass or just reuse if we don't modify it permanently
    # Kahn's modifies in_deg, so we need to be careful or recompute.
    # Let's just do the topo sort once and store the order.
    in_deg_copy = in_deg[:]
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in succs[node]:
            in_deg_copy[s] -= 1
            if in_deg_copy[s] == 0:
                queue.append(s)
    
    if len(topo_order) < n:
        # Cycle detected, fallback to original order
        topo_order = list(range(n))

    # Forward Pass: Earliest Start (ES) and Earliest Finish (EF)
    ES = np.zeros(n, dtype=float)
    EF = np.zeros(n, dtype=float)
    for node in topo_order:
        if in_deg[node] > 0: # Has predecessors
            # ES is max EF of predecessors
            max_pred_ef = 0.0
            for p in (preds[node] if preds is not None else []):
                if EF[p] > max_pred_ef:
                    max_pred_ef = EF[p]
            ES[node] = max_pred_ef
        else:
            ES[node] = 0.0
        EF[node] = ES[node] + durations[node]
    
    # Project Makespan (ignoring resources)
    project_end = max(EF) if n > 0 else 0.0

    # Backward Pass: Latest Finish (LF) and Latest Start (LS)
    LF = np.zeros(n, dtype=float)
    LS = np.zeros(n, dtype=float)
    # Initialize LF for sinks to project_end
    # We process in reverse topological order
    for node in reversed(topo_order):
        if not succs[node]:
            LF[node] = project_end
        else:
            # LF is min LS of successors
            min_succ_ls = float('inf')
            for s in succs[node]:
                if LS[s] < min_succ_ls:
                    min_succ_ls = LS[s]
            LF[node] = min_succ_ls
        LS[node] = LF[node] - durations[node]
        
        # Ensure LS is not negative (though theoretically it shouldn't be if graph is valid)
        if LS[node] < 0:
            LS[node] = 0.0

    # Calculate Total Float (Slack)
    # TF = LS - ES
    TF = LS - ES
    
    # Normalize Total Float to [0, 1]
    # We want Low TF to have High Priority.
    # So we will use (1 - TF_norm) as the primary score component.
    tf_min = TF.min() if n > 0 else 0.0
    tf_max = TF.max() if n > 0 else 0.0
    
    if tf_max - tf_min < 1e-9:
        tf_norm = np.zeros(n, dtype=float) # All have same slack, so 0 slack norm? 
        # Actually if all same, they are all equally critical.
        # Let's set tf_norm to 0 for all, so 1 - 0 = 1 for all.
    else:
        tf_norm = (TF - tf_min) / (tf_max - tf_min)
    
    # Slack Score: Higher is better (meaning lower slack)
    slack_score = 1.0 - tf_norm
    
    # Resource Demand Intensity
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    demand_score = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands, dtype=float)
        caps_arr = np.asarray(capacities, dtype=float)
        
        if demands_arr.ndim == 2:
            # demands_arr[i, k]
            if len(caps_arr) == demands_arr.shape[1]:
                norm_dem = demands_arr / np.maximum(caps_arr, 1e-9)
                # Use max normalized demand per activity
                demand_score = np.max(norm_dem, axis=1)
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

    # Normalize Demand Score
    def normalize(x):
        x = np.asarray(x, dtype=float)
        if x.max() - x.min() < 1e-9:
            return np.ones_like(x) * 0.5
        return (x - x.min()) / (x.max() - x.min())
    
    dem_norm = normalize(demand_score)
    
    # Combine Scores
    # Slack is the dominant factor for makespan.
    # Demand is a tie-breaker / secondary factor to handle resource constraints.
    # Weighting: 0.8 Slack, 0.2 Demand
    score = 0.8 * slack_score + 0.2 * dem_norm
    
    return score
# EVOLVE-BLOCK-END
