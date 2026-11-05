"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict
import time

def priority(instance):
    """
    Computes priority scores for activities using a refined hybrid heuristic:
    1. Critical path proximity via forward/backward pass (total float)
    2. Resource pressure via log-sum-exp surrogate for max demand ratio
    3. Duration weighting - longer activities benefit more from early scheduling
    
    The final score prioritizes activities that are both on/near the critical
    path AND have high aggregate resource demands, preventing bottlenecks.
    """
    # Extract instance data
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([])
    
    # Get predecessors
    pred_list = instance.get("predecessors", None)
    if pred_list is None:
        pred_list = [[] for _ in range(n)]
    
    # Get resource data
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    # Build successor list
    succ = defaultdict(list)
    for i in range(n):
        for p in pred_list[i]:
            succ[p].append(i)
    
    # Topological order via Kahn's algorithm
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred_list[i]:
            in_degree[i] += 1
    
    from queue import Queue
    q = Queue()
    for i in range(n):
        if in_degree[i] == 0:
            q.put(i)
    
    topo_order = []
    while not q.empty():
        u = q.get()
        topo_order.append(u)
        for v in succ[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.put(v)
    
    # If cycle detected (shouldn't happen in valid RCPSP), fall back
    if len(topo_order) < n:
        topo_order = list(range(n))
    
    # Forward pass: earliest start (ES) and earliest finish (EF)
    es = np.zeros(n, dtype=float)
    ef = np.zeros(n, dtype=float)
    for u in topo_order:
        max_pred_ef = 0.0
        for p in pred_list[u]:
            if ef[p] > max_pred_ef:
                max_pred_ef = ef[p]
        es[u] = max_pred_ef
        ef[u] = es[u] + durations[u]
    
    # Backward pass: latest finish (LF) and latest start (LS)
    project_end = np.max(ef) if n > 0 else 0.0
    lf = np.full(n, project_end, dtype=float)
    ls = np.full(n, project_end, dtype=float)
    
    for u in reversed(topo_order):
        # Latest finish of u = min(latest start of all successors)
        min_succ_ls = project_end
        for v in succ[u]:
            if ls[v] < min_succ_ls:
                min_succ_ls = ls[v]
        lf[u] = min_succ_ls
        ls[u] = lf[u] - durations[u]
    
    # Total float = LS - ES
    total_float = ls - es
    
    # Normalize total float: activities with 0 float are on critical path (highest priority)
    tf_min = np.min(total_float)
    tf_max = np.max(total_float)
    if tf_max > tf_min:
        # Invert: low float -> high score
        criticality = 1.0 - (total_float - tf_min) / (tf_max - tf_min)
    else:
        criticality = np.ones(n, dtype=float)
    
    # Compute resource contention
    if demands is not None and capacities is not None:
        demands = np.asarray(demands, dtype=float)
        capacities = np.asarray(capacities, dtype=float)
        
        if demands.ndim == 1:
            demands = demands.reshape(-1, 1)
            capacities = capacities.reshape(-1)
        
        safe_capacities = np.where(capacities > 0, capacities, 1.0)
        
        # Compute ratio for each activity-resource pair
        ratios = demands / safe_capacities  # shape: (n, r)
        
        # Log-sum-exp for smooth max approximation
        # Use temperature parameter for smoothness control
        temp = 0.3  # Lower temp = sharper max approximation
        
        # Numerically stable log-sum-exp
        ratios_scaled = ratios / temp
        max_ratios = np.max(ratios_scaled, axis=1, keepdims=True)
        exp_terms = np.exp(ratios_scaled - max_ratios)
        log_sum = np.sum(exp_terms, axis=1)
        lse = max_ratios[:, 0] + np.log(log_sum)
        
        # Scale back
        resource_contention = lse * temp
        
        # Also compute weighted sum of ratios as a secondary measure
        # Weight by how "tight" each resource is globally
        avg_ratios = np.mean(ratios, axis=0)  # average ratio per resource
        # Normalize weights
        if np.max(avg_ratios) > 0:
            res_weights = avg_ratios / np.max(avg_ratios)
        else:
            res_weights = np.ones_like(avg_ratios)
        
        weighted_contention = ratios @ res_weights
    else:
        resource_contention = np.zeros(n, dtype=float)
        weighted_contention = np.zeros(n, dtype=float)
    
    # Normalize resource metrics
    rc_min, rc_max = np.min(resource_contention), np.max(resource_contention)
    if rc_max > rc_min:
        rc_norm = (resource_contention - rc_min) / (rc_max - rc_min)
    else:
        rc_norm = np.zeros(n, dtype=float)
    
    wc_min, wc_max = np.min(weighted_contention), np.max(weighted_contention)
    if wc_max > wc_min:
        wc_norm = (weighted_contention - wc_min) / (wc_max - wc_min)
    else:
        wc_norm = np.zeros(n, dtype=float)
    
    # Normalize duration
    dur_min, dur_max = np.min(durations), np.max(durations)
    if dur_max > dur_min:
        dur_norm = (durations - dur_min) / (dur_max - dur_min)
    else:
        dur_norm = np.zeros(n, dtype=float)
    
    # Combine: criticality (0.45), smooth resource contention (0.25), weighted contention (0.15), duration (0.15)
    weight_crit = 0.45
    weight_rc = 0.25
    weight_wc = 0.15
    weight_dur = 0.15
    
    scores = weight_crit * criticality + weight_rc * rc_norm + weight_wc * wc_norm + weight_dur * dur_norm
    
    # Scale to a meaningful range based on project duration
    # This ensures scores are in a consistent scale
    if project_end > 0:
        scores = scores * project_end
    else:
        scores = scores
    
    return scores

# EVOLVE-BLOCK-END
