"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

_TIME_LIMIT = 145.0
_START_TIME = None

def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Heuristic: Bottom Level criticality with smooth resource-contention penalty.
    Uses log-sum-exp for soft-max in CPM calculations and a smooth surrogate
    for resource contention to avoid hard discontinuities in priority ordering.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Emergency fallback
        durations = np.asarray(instance["durations"], dtype=float)
        if len(durations) == 0: return np.array([])
        return durations / (np.max(durations) + 1e-9)

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([])

    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    preds = instance.get("predecessors", [None] * n)

    # 1. Build graph
    pred_lists = []
    for p in preds:
        if p is None:
            pred_lists.append([])
        elif isinstance(p, (list, tuple, np.ndarray)):
            pred_lists.append([x for x in p if x is not None])
        else:
            pred_lists.append([p])
            
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in pred_lists[i]:
            if 0 <= p < n:
                successors[p].append(i)
                in_degree[i] += 1
            
    # Topological Sort
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    top_order = []
    while queue:
        u = queue.popleft()
        top_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(top_order) < n:
        top_order = list(range(n))

    # 2. Compute Bottom Level (Longest path from node to end)
    # Using soft-max for smoothness: log-sum-exp approximation
    bottom_level = np.zeros(n)
    for u in reversed(top_order):
        if successors[u]:
            succ_bl = np.array([bottom_level[v] for v in successors[u]])
            # Soft-max with temperature parameter
            temp = 1.0
            max_val = np.max(succ_bl)
            if max_val > 0:
                soft_max = max_val + temp * np.log(np.sum(np.exp((succ_bl - max_val) / temp)))
            else:
                soft_max = 0.0
            bottom_level[u] = durations[u] + soft_max
        else:
            bottom_level[u] = durations[u]

    # 3. Compute Resource Pressure with smooth contention metric
    rp = np.zeros(n)
    if demands is not None and capacities is not None:
        capacities_arr = np.asarray(capacities, dtype=float)
        if len(capacities_arr) == 0:
            capacities_arr = np.array([1.0])
        else:
            capacities_arr = np.where(capacities_arr > 0, capacities_arr, 1.0)
            
        # Process demands
        n_res = len(capacities_arr)
        # Initialize demand matrix
        d_matrix = np.zeros((n, n_res), dtype=float)
        
        if isinstance(demands, list):
            for i in range(n):
                if i < len(demands):
                    d_i = demands[i]
                    if isinstance(d_i, dict):
                        for r, val in d_i.items():
                            if r < n_res:
                                d_matrix[i, r] = val
                    elif isinstance(d_i, (list, tuple, np.ndarray)):
                        d_arr = np.asarray(d_i, dtype=float)
                        if d_arr.ndim == 1:
                            min_len = min(len(d_arr), n_res)
                            d_matrix[i, :min_len] = d_arr[:min_len]
                    else:
                        # Scalar?
                        if n_res > 0:
                            d_matrix[i, 0] = float(d_i)
        else:
            # Try converting whole thing to array
            try:
                d_mat_full = np.asarray(demands, dtype=float)
                if d_mat_full.shape[0] == n:
                    if d_mat_full.ndim == 1:
                         d_matrix[:, 0] = d_mat_full
                    else:
                        min_res = min(d_mat_full.shape[1], n_res)
                        d_matrix[:, :min_res] = d_mat_full[:, :min_res]
            except:
                pass

        # Calculate smooth resource contention penalty
        # For each activity i and resource r, compute a smooth measure of how much
        # it competes with others. Use log-sum-exp over all activities sharing resource r.
        for r in range(n_res):
            cap_r = capacities_arr[r]
            d_r = d_matrix[:, r]
            # Only consider activities that use this resource
            active = d_r > 0
            if np.sum(active) > 0:
                # Smooth contention: for each active activity, compute log-sum-exp of 
                # demand ratios of all active activities, minus its own ratio
                ratios = d_r[active] / cap_r
                log_sum_exp = np.log(np.sum(np.exp(ratios)))
                # Contention score for each active activity: how much others contribute
                # to the total resource usage
                for idx, i in enumerate(np.where(active)[0]):
                    own_ratio = d_matrix[i, r] / cap_r
                    # Smooth penalty: exp(log_sum_exp - own_ratio) approximates sum of others' exp
                    if own_ratio > 0:
                        rp[i] += np.exp(log_sum_exp - own_ratio) - 1.0  # subtract 1 to remove self
                    else:
                        rp[i] += np.exp(log_sum_exp)

    # 4. Normalize Metrics
    # Bottom Level
    bl_max = np.max(bottom_level) if np.max(bottom_level) > 0 else 1.0
    bl_min = np.min(bottom_level)
    bl_range = bl_max - bl_min
    if bl_range > 0:
        bl_norm = (bottom_level - bl_min) / bl_range
    else:
        bl_norm = np.ones(n)

    # Resource Pressure
    rp_max = np.max(rp) if np.max(rp) > 0 else 1.0
    rp_min = np.min(rp)
    rp_range = rp_max - rp_min
    if rp_range > 0:
        rp_norm = (rp - rp_min) / rp_range
    else:
        rp_norm = np.zeros(n)

    # Duration
    dur_max = np.max(durations) if np.max(durations) > 0 else 1.0
    dur_min = np.min(durations)
    dur_range = dur_max - dur_min
    if dur_range > 0:
        dur_norm = (durations - dur_min) / dur_range
    else:
        dur_norm = np.ones(n)

    # Compute Total Float (TF) for each activity using soft-min for smoothness
    es = np.zeros(n)
    ef = np.zeros(n)
    for u in top_order:
        if pred_lists[u]:
            pred_ef = np.array([ef[p] for p in pred_lists[u]])
            # Soft-min with temperature
            temp = 1.0
            min_val = np.min(pred_ef)
            if min_val > 0:
                soft_min = min_val - temp * np.log(np.sum(np.exp((min_val - pred_ef) / temp)))
            else:
                soft_min = 0.0
            es[u] = soft_min
        else:
            es[u] = 0.0
        ef[u] = es[u] + durations[u]
        
    cp_length = np.max(ef) if n > 0 else 0.0
    
    ls = np.zeros(n)
    lf = np.zeros(n)
    for u in reversed(top_order):
        if successors[u]:
            succ_ls = np.array([ls[v] for v in successors[u]])
            # Soft-min
            temp = 1.0
            min_val = np.min(succ_ls)
            if min_val > 0:
                soft_min = min_val - temp * np.log(np.sum(np.exp((min_val - succ_ls) / temp)))
            else:
                soft_min = cp_length
            lf[u] = soft_min
        else:
            lf[u] = cp_length
        ls[u] = lf[u] - durations[u]
        
    tf = ls - es
    
    tf_max = np.max(tf) if np.max(tf) > 0 else 1.0
    tf_min = np.min(tf)
    tf_range = tf_max - tf_min
    if tf_range > 0:
        tf_norm = (tf - tf_min) / tf_range
    else:
        tf_norm = np.zeros(n)
        
    # Combine scores: criticality (BL) as main driver, with smooth resource contention penalty
    # and total float as secondary signal
    w_bl = 0.70
    w_rp = 0.15
    w_tf = 0.15
    
    # Smooth combination: BL boosted slightly by duration, penalized by resource contention
    # and total float
    scores = w_bl * bl_norm + 0.05 * dur_norm - w_rp * rp_norm - w_tf * tf_norm

    # Ensure scores are in a reasonable range
    score_max = np.max(scores) if np.max(scores) > 0 else 1.0
    score_min = np.min(scores)
    score_range = score_max - score_min
    if score_range > 0:
        scores = (scores - score_min) / score_range
    else:
        scores = np.ones(n)

    return scores
# EVOLVE-BLOCK-END
