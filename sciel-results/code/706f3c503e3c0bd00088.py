"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Compute priority scores for each activity in an RCPSP instance.
    Higher priority = scheduled earlier in serial SGS.
    
    Strategy: Combine critical path position (low total float = high priority),
    resource demand pressure, and successor count to form a composite priority.
    """
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Parse predecessors: instance["predecessors"] is list of lists or list of arrays
    pred = instance.get("predecessors", [])
    # Ensure pred is a list of lists of ints
    if isinstance(pred, np.ndarray):
        pred = pred.tolist()
    pred_lists = []
    for p in pred:
        if isinstance(p, (list, np.ndarray)):
            pred_lists.append([int(x) for x in p])
        else:
            pred_lists.append([])
    
    # Build successors
    succ = [[] for _ in range(n)]
    for i, plist in enumerate(pred_lists):
        for p in plist:
            succ[p].append(i)
    
    # Demands and capacities
    demands = np.asarray(instance.get("demands", instance.get("resource_demands", np.zeros((n, 4)))), dtype=float)
    capacities = np.asarray(instance.get("capacities", instance.get("resource_capacities", np.array([10, 10, 10, 10]))), dtype=float)
    
    if demands.ndim == 1:
        demands = demands.reshape(n, -1)
    
    # Forward pass: earliest start times
    es = np.zeros(n)
    # Topological order
    in_degree = [len(plist) for plist in pred_lists]
    from collections import deque
    topo = []
    q = deque([i for i in range(n) if in_degree[i] == 0])
    in_deg_copy = in_degree[:]
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succ[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)
    
    # Forward pass
    for u in topo:
        for v in succ[u]:
            if es[v] < es[u] + durations[u]:
                es[v] = es[u] + durations[u]
    
    # Project makespan estimate (forward)
    makespan_fwd = max(es + durations) if n > 0 else 0
    
    # Backward pass: latest start times
    ls = np.full(n, makespan_fwd)
    for u in reversed(topo):
        for v in succ[u]:
            if ls[u] > ls[v] - durations[u]:
                ls[u] = ls[v] - durations[u]
    
    # Total float
    total_float = ls - es
    
    # Priority components
    # 1. Criticality: inverse of total float (activities with 0 float are most critical)
    #    Use 1/(float + 1) to avoid division by zero, scaled
    crit_priority = 1.0 / (total_float + 1.0)
    
    # 2. Resource pressure: for each activity, how much it stresses resources
    #    Compute demand/capacity ratio, weighted by duration
    if demands.shape[1] > 0 and capacities.size > 0:
        # Normalize demands by capacities
        cap_safe = np.maximum(capacities, 1.0)
        demand_ratio = demands / cap_safe[np.newaxis, :]  # (n, R)
        # Weight by duration: longer activities that use more resources are more important
        res_pressure = np.sum(demand_ratio * durations[:, np.newaxis], axis=1) / max(demands.shape[1], 1)
    else:
        res_pressure = np.zeros(n)
    
    # 3. Successor count: activities that unlock many others
    succ_count = np.array([len(s) for s in succ], dtype=float)
    max_succ = max(succ_count.max(), 1.0)
    succ_priority = succ_count / max_succ
    
    # 4. Duration-based: longer activities slightly preferred (to reduce idle time)
    max_dur = max(durations.max(), 1.0)
    dur_priority = durations / max_dur
    
    # Composite priority: weighted sum
    # Critical path position is most important
    w_crit = 0.5
    w_res = 0.2
    w_succ = 0.15
    w_dur = 0.15
    
    # Normalize each component to [0, 1]
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-9:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)
    
    p_crit = normalize(crit_priority)
    p_res = normalize(res_pressure)
    p_succ = normalize(succ_priority)
    p_dur = normalize(dur_priority)
    
    priority_scores = (w_crit * p_crit + w_res * p_res + w_succ * p_succ + w_dur * p_dur)
    
    # Time guard
    if time.time() - t0 > DEADLINE:
        return priority_scores
    
    return priority_scores
# EVOLVE-BLOCK-END
