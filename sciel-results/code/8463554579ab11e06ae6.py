"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each activity using a resource-weighted criticality measure.
    
    Strategy:
    1. Compute static CPM (Critical Path Method) latest start times.
    2. Compute demand-to-capacity ratios for each activity-resource pair.
    3. Combine these into a single priority score that favors activities on
       the critical path that also consume significant resources.
    """
    # Get instance data
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Get predecessors - handle different possible formats
    preds = instance.get("predecessors", None)
    if preds is None:
        preds = instance.get("precedences", None)
    
    # Get demands and capacities
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    # Build predecessor/successor lists
    succ = [[] for _ in range(n)]
    pred = [[] for _ in range(n)]
    
    if preds is not None:
        # preds could be a list of lists, or a dict, or a list of arrays
        if isinstance(preds, dict):
            for i, plist in preds.items():
                i = int(i)
                for j in plist:
                    j = int(j)
                    succ[j].append(i)
                    pred[i].append(j)
        elif isinstance(preds, (list, tuple)):
            for i in range(n):
                if i < len(preds):
                    for j in preds[i]:
                        j = int(j)
                        succ[j].append(i)
                        pred[i].append(j)
        else:
            # Try treating as numpy array
            try:
                preds_arr = np.asarray(preds)
                if preds_arr.ndim == 2:
                    for i in range(n):
                        if i < preds_arr.shape[0]:
                            for j in preds_arr[i]:
                                if j is not None and j >= 0:
                                    j = int(j)
                                    succ[j].append(i)
                                    pred[i].append(j)
            except:
                pass
    
    # Compute earliest start/finish times (forward pass)
    es = np.zeros(n, dtype=float)
    ef = np.zeros(n, dtype=float)
    
    # Topological order
    in_degree = np.array([len(pred[i]) for i in range(n)], dtype=int)
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succ[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If topological sort didn't include all nodes (cycle?), fall back
    if len(topo_order) < n:
        topo_order = list(range(n))
    
    for u in topo_order:
        for v in succ[u]:
            if ef[u] > es[v]:
                es[v] = ef[u]
        ef[u] = es[u] + durations[u]
    
    project_duration = max(ef) if n > 0 else 0.0
    
    # Compute latest start/finish times (backward pass)
    ls = np.zeros(n, dtype=float)
    lf = np.zeros(n, dtype=float)
    
    # Initialize lf for activities with no successors
    for i in range(n):
        if len(succ[i]) == 0:
            lf[i] = project_duration
        else:
            lf[i] = project_duration  # will be refined
    
    # Backward pass in reverse topological order
    for u in reversed(topo_order):
        if len(succ[u]) > 0:
            lf[u] = min(lf[v] for v in succ[u])
        ls[u] = lf[u] - durations[u]
    
    # Slack (total float)
    slack = ls - es
    
    # Compute demand-to-capacity ratio for each activity
    # If demands/capacities are available
    demand_ratio = np.zeros(n, dtype=float)
    
    if demands is not None and capacities is not None:
        try:
            demands_arr = np.asarray(demands, dtype=float)
            capacities_arr = np.asarray(capacities, dtype=float)
            
            # demands_arr shape: (n_activities, n_resources) or (n_resources, n_activities)
            # capacities_arr shape: (n_resources,)
            if demands_arr.ndim == 2:
                if demands_arr.shape[0] == n:
                    # (n_activities, n_resources)
                    # For each activity, max over resources of demand/capacity
                    # Handle zero capacity
                    safe_cap = np.where(capacities_arr > 0, capacities_arr, 1.0)
                    ratios = demands_arr / safe_cap[np.newaxis, :]
                    demand_ratio = np.max(ratios, axis=1)
                elif demands_arr.shape[1] == n:
                    # (n_resources, n_activities)
                    safe_cap = np.where(capacities_arr > 0, capacities_arr, 1.0)
                    ratios = demands_arr / safe_cap[:, np.newaxis]
                    demand_ratio = np.max(ratios, axis=0)
            else:
                # 1D demands
                safe_cap = np.where(capacities_arr > 0, capacities_arr, 1.0)
                demand_ratio = demands_arr / safe_cap
        except:
            demand_ratio = np.zeros(n, dtype=float)
    
    # Normalize slack to [0, 1] range (lower slack = higher criticality)
    max_slack = np.max(slack) if n > 0 else 1.0
    if max_slack > 0:
        criticality = 1.0 - (slack / max_slack)
    else:
        criticality = np.ones(n, dtype=float)
    
    # Normalize duration
    max_dur = np.max(durations) if n > 0 else 1.0
    if max_dur > 0:
        dur_norm = durations / max_dur
    else:
        dur_norm = np.ones(n, dtype=float)
    
    # Combine: weighted sum of criticality, duration, and demand ratio
    # Higher score = higher priority
    # Criticality is most important, then demand ratio, then duration
    score = 1.0 * criticality + 0.5 * demand_ratio + 0.3 * dur_norm
    
    return score
# EVOLVE-BLOCK-END
