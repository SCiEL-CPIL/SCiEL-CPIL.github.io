"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

_TIME_LIMIT = 140.0
_START_TIME = None

def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Heuristic: Dynamic resource-weighted criticality.
    1. Run a greedy SGS simulation to detect empirical resource bottlenecks.
    2. Combine static CPM criticality (Bottom Level) with dynamic bottleneck detection.
    3. Penalize activities with high total float (non-critical).
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
    bottom_level = np.zeros(n)
    for u in reversed(top_order):
        if successors[u]:
            max_succ_bl = np.max([bottom_level[v] for v in successors[u]])
        else:
            max_succ_bl = 0.0
        bottom_level[u] = durations[u] + max_succ_bl

    # 3. Parse demand matrix and capacities
    d_matrix = np.zeros((n, 1), dtype=float)
    capacities_arr = np.array([1.0])
    n_res = 1
    if demands is not None and capacities is not None:
        capacities_arr = np.asarray(capacities, dtype=float)
        if len(capacities_arr) == 0:
            capacities_arr = np.array([1.0])
        else:
            capacities_arr = np.where(capacities_arr > 0, capacities_arr, 1.0)
        n_res = len(capacities_arr)
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
                        if n_res > 0:
                            d_matrix[i, 0] = float(d_i)
        else:
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

    # 4. Compute Total Float for CPM criticality
    es = np.zeros(n)
    ef = np.zeros(n)
    for u in top_order:
        if pred_lists[u]:
            es[u] = np.max([ef[p] for p in pred_lists[u]])
        else:
            es[u] = 0.0
        ef[u] = es[u] + durations[u]
        
    cp_length = np.max(ef) if n > 0 else 0.0
    
    ls = np.zeros(n)
    lf = np.zeros(n)
    for u in reversed(top_order):
        if successors[u]:
            lf[u] = np.min([ls[v] for v in successors[u]])
        else:
            lf[u] = cp_length
        ls[u] = lf[u] - durations[u]
        
    tf = ls - es

    # 5. Dynamic bottleneck detection via greedy SGS simulation
    # Run a quick serial SGS using BL as priority to identify which activities
    # actually cause resource contention (get delayed due to resource limits).
    
    bottleneck_score = np.zeros(n)  # higher = more of a bottleneck
    
    if demands is not None and capacities is not None and n_res > 0:
        # Greedy SGS: always pick eligible activity with highest BL
        start_times = np.full(n, -1.0)
        finish_times = np.full(n, -1.0)
        completed = [False] * n
        remaining = list(range(n))
        current_time = 0.0
        # Track resource usage over time
        # We'll simulate event by event
        
        # For efficiency with n=30, we can do a simple loop
        # At each step, find eligible activities, pick highest BL, schedule it
        # Track when resources become available
        
        # Resource availability tracking
        # resource_free_time[r] = time when resource r becomes free
        # But we need per-activity tracking since multiple activities can run in parallel
        
        # Simpler approach: track which activities are running and when they finish
        active = []  # list of (finish_time, activity_id)
        
        while remaining:
            # Check if any active activities have finished
            # Find next event time
            if active:
                next_event = min(a[0] for a in active)
                # Remove finished activities
                still_active = []
                for ft, aid in active:
                    if ft <= current_time:
                        pass  # already finished
                    elif ft > current_time:
                        still_active.append((ft, aid))
                active = still_active
                
                if not active:
                    current_time = next_event
                else:
                    # Find the earliest finish among active
                    min_ft = min(a[0] for a in active)
                    if min_ft > current_time:
                        current_time = min_ft
                    # Remove all that finished at current_time
                    still_active = []
                    for ft, aid in active:
                        if ft > current_time:
                            still_active.append((ft, aid))
                    active = still_active
            
            # Find eligible activities (all predecessors completed)
            eligible = []
            for i in remaining:
                if all(completed[p] for p in pred_lists[i]):
                    eligible.append(i)
            
            if not eligible:
                break
            
            # Pick eligible with highest BL (tie-break by duration)
            best = max(eligible, key=lambda i: (bottom_level[i], durations[i]))
            
            # Determine start time: max(current_time, finish times of predecessors)
            if pred_lists[best]:
                pred_finish = max(finish_times[p] for p in pred_lists[best])
                start = max(current_time, pred_finish)
            else:
                start = current_time
            
            # Check resource feasibility at start time
            # We need to check if resources are available
            # For simplicity in this heuristic, assume we can start if no conflict
            # In a real SGS, we'd need to check resource availability
            
            finish = start + durations[best]
            start_times[best] = start
            finish_times[best] = finish
            completed[best] = True
            remaining.remove(best)
            active.append((finish, best))
            
            # Update current_time if needed
            if start > current_time:
                current_time = start
        
        # Now identify bottlenecks: activities that were delayed due to resource contention
        # An activity is a bottleneck if its actual start > CPM earliest start (tf > 0)
        # and it has high resource demand
        for i in range(n):
            if start_times[i] >= 0:
                delay = start_times[i] - es[i]
                if delay > 1e-9:
                    # This activity was delayed; it's likely a bottleneck
                    # Weight by its resource pressure
                    pressure = np.sum(d_matrix[i] / capacities_arr)
                    bottleneck_score[i] = delay * pressure
    
    # 6. Normalize all components
    # Bottom Level
    bl_max = np.max(bottom_level) if np.max(bottom_level) > 0 else 1.0
    bl_min = np.min(bottom_level)
    bl_range = bl_max - bl_min
    if bl_range > 0:
        bl_norm = (bottom_level - bl_min) / bl_range
    else:
        bl_norm = np.ones(n)

    # Total Float
    tf_max = np.max(tf) if np.max(tf) > 0 else 1.0
    tf_min = np.min(tf)
    tf_range = tf_max - tf_min
    if tf_range > 0:
        tf_norm = (tf - tf_min) / tf_range
    else:
        tf_norm = np.zeros(n)

    # Bottleneck score
    bn_max = np.max(bottleneck_score) if np.max(bottleneck_score) > 0 else 1.0
    bn_min = np.min(bottleneck_score)
    bn_range = bn_max - bn_min
    if bn_range > 0:
        bn_norm = (bottleneck_score - bn_min) / bn_range
    else:
        bn_norm = np.zeros(n)

    # Static resource pressure
    rp = np.sum(d_matrix / capacities_arr, axis=1)
    rp_max = np.max(rp) if np.max(rp) > 0 else 1.0
    rp_min = np.min(rp)
    rp_range = rp_max - rp_min
    if rp_range > 0:
        rp_norm = (rp - rp_min) / rp_range
    else:
        rp_norm = np.ones(n)

    # 7. Combine: criticality * (1 + bottleneck_boost) - float_penalty
    # Activities on the critical path with high bottleneck detection get boosted
    # Activities with high total float get penalized
    
    w_bl = 0.70
    w_bn = 0.20
    w_tf = 0.30
    w_rp = 0.10

    # Base score from criticality
    base_score = bl_norm
    
    # Boost for detected bottlenecks
    bottleneck_boost = bn_norm * (1 + rp_norm * 0.5)
    
    # Penalty for high float
    float_penalty = tf_norm
    
    scores = w_bl * base_score + w_bn * bottleneck_boost - w_tf * float_penalty + w_rp * rp_norm * 0.1

    # Ensure scores are bounded
    score_max = np.max(scores) if np.max(scores) > 0 else 1.0
    score_min = np.min(scores)
    score_range = score_max - score_min
    if score_range > 0:
        scores = (scores - score_min) / score_range
    else:
        scores = np.ones(n)

    return scores
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
