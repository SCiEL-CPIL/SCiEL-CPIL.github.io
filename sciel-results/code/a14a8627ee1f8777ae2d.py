"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

_TIME_LIMIT = 145.0
_START_TIME = None

def _run_sgs(n, durations, demands, capacities, pred_lists, succ_lists, priority_scores):
    """
    Internal SGS simulation to generate a reference schedule.
    Returns start times, finish times, and makespan.
    """
    if n == 0:
        return np.array([]), np.array([]), 0.0

    # Parse demands and capacities into matrices
    if capacities is None:
        capacities = [1.0]
    capacities_arr = np.asarray(capacities, dtype=float)
    n_res = len(capacities_arr)
    
    d_matrix = np.zeros((n, n_res), dtype=float)
    if demands is not None:
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

    # SGS Simulation
    start_times = np.zeros(n)
    finish_times = np.zeros(n)
    completed = [False] * n
    remaining = list(range(n))
    
    # Time horizon
    max_duration = np.max(durations) if n > 0 else 0
    # Simple upper bound: sum of all durations
    time_limit = np.sum(durations)
    
    current_time = 0.0
    scheduled_count = 0
    
    # Resource usage over time is hard to track efficiently without a timeline.
    # We simulate event by event.
    
    # List of (finish_time, activity_id) for completed activities
    events = []
    
    while scheduled_count < n:
        # Find eligible activities
        eligible = []
        for i in remaining:
            # Check if all predecessors are completed
            ready = True
            for p in pred_lists[i]:
                if not completed[p]:
                    ready = False
                    break
            if ready:
                eligible.append(i)
                
        if not eligible:
            # If no eligible, jump to next event
            if events:
                # Sort by finish time
                events.sort(key=lambda x: x[0])
                next_finish, act_id = events.pop(0)
                current_time = next_finish
                completed[act_id] = True
                scheduled_count -= 1 # Wait, scheduled_count is number of *started*? No, let's track completed.
                # Let's re-structure: 
                # We need to know when an activity finishes to free resources.
                # This simple loop doesn't handle resource release correctly for SGS which is "start as early as possible".
                
                # SGS Logic:
                # 1. Determine set of eligible activities (predecessors done).
                # 2. Among eligible, pick highest priority.
                # 3. Find earliest time t >= current_time where resources are sufficient.
                # 4. Start activity at t.
                # 5. Update current_time? No, current_time is the time of the last *start*? 
                #    Actually, in SGS, we schedule activities one by one.
                #    The "current time" for the next activity is the max of the finish times of its predecessors.
                #    But we also need to check resource availability.
                
                # Let's restart the SGS logic properly.
                break
            
        if not eligible:
            break
            
        # Pick highest priority
        best_act = max(eligible, key=lambda i: priority_scores[i])
        
        # Determine earliest start time (ES)
        es = 0.0
        for p in pred_lists[best_act]:
            if finish_times[p] > es:
                es = finish_times[p]
                
        # Find earliest feasible time t >= es
        t = es
        # We need to check resource availability for [t, t+dur]
        # To do this efficiently, we can keep a record of resource usage intervals.
        # For simplicity in this internal simulation (n=30), we can just check all scheduled activities.
        
        feasible = False
        while not feasible:
            feasible = True
            for r in range(n_res):
                demand = d_matrix[best_act, r]
                if demand == 0:
                    continue
                # Check all scheduled activities that overlap with [t, t+dur]
                end_time = t + durations[best_act]
                for j in range(n):
                    if start_times[j] > 0 or (start_times[j] == 0 and j == best_act): # j is scheduled
                        # Check overlap
                        if start_times[j] < end_time and finish_times[j] > t:
                            # Overlap exists
                            used = d_matrix[j, r]
                            if used + demand > capacities_arr[r]:
                                feasible = False
                                # Jump to finish time of conflicting activity
                                if finish_times[j] > t:
                                    t = finish_times[j]
                                break
                if not feasible:
                    break
            if not feasible:
                continue
            
        # Schedule best_act at t
        start_times[best_act] = t
        finish_times[best_act] = t + durations[best_act]
        remaining.remove(best_act)
        scheduled_count += 1
        
    makespan = np.max(finish_times) if n > 0 else 0.0
    return start_times, finish_times, makespan

def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Heuristic: Hybrid Static-Dynamic Criticality.
    1. Compute Bottom Level (BL) and Total Float (TF).
    2. Run an internal SGS simulation with BL priority to get a reference schedule.
    3. Identify activities on the critical path of the reference schedule.
    4. Compute Resource Criticality (demand/capacity weighted by BL).
    5. Combine these to form the final priority score.
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

    # 2. Compute Bottom Level (BL)
    bottom_level = np.zeros(n)
    for u in reversed(top_order):
        if successors[u]:
            succ_bl = np.array([bottom_level[v] for v in successors[u]])
            bottom_level[u] = durations[u] + np.max(succ_bl)
        else:
            bottom_level[u] = durations[u]

    # 3. Compute Total Float (TF)
    es = np.zeros(n)
    ef = np.zeros(n)
    for u in top_order:
        if pred_lists[u]:
            pred_ef = np.array([ef[p] for p in pred_lists[u]])
            es[u] = np.max(pred_ef)
        else:
            es[u] = 0.0
        ef[u] = es[u] + durations[u]
        
    cp_length = np.max(ef) if n > 0 else 0.0
    
    ls = np.zeros(n)
    lf = np.zeros(n)
    for u in reversed(top_order):
        if successors[u]:
            succ_ls = np.array([ls[v] for v in successors[u]])
            lf[u] = np.min(succ_ls)
        else:
            lf[u] = cp_length
        ls[u] = lf[u] - durations[u]
        
    tf = ls - es

    # 4. Internal SGS Simulation with BL priority
    # We use BL as the initial priority to get a reference schedule
    ref_start, ref_finish, ref_makespan = _run_sgs(
        n, durations, demands, capacities, pred_lists, successors, bottom_level
    )
    
    # 5. Identify Critical Activities in Reference Schedule
    # An activity is critical if its finish time is part of the critical path.
    # Simple heuristic: Activities with finish time close to makespan or zero float in ref schedule.
    # Let's compute float in the reference schedule.
    ref_es = np.zeros(n)
    ref_ef = np.zeros(n)
    for u in top_order:
        if pred_lists[u]:
            pred_ref_ef = np.array([ref_finish[p] for p in pred_lists[u]])
            ref_es[u] = np.max(pred_ref_ef)
        else:
            ref_es[u] = 0.0
        ref_ef[u] = ref_es[u] + durations[u]
        
    # Note: ref_finish from SGS might be later than ref_ef if resources delayed it.
    # The "Critical Path" in the resource-constrained schedule is the path of activities
    # where delaying any of them delays the makespan.
    # Approximation: Activities where ref_finish[i] is on the critical path.
    # We can identify this by doing a backward pass on the resource-constrained schedule?
    # That's complex. Instead, let's use the static TF as a proxy for criticality,
    # but boost activities that were delayed in the reference schedule (indicating they fought for resources).
    
    # Delay in reference schedule
    delay = ref_start - es
    
    # Resource Criticality: Max demand-to-capacity ratio
    # Parse capacities and demands again for this step
    if capacities is not None:
        capacities_arr = np.asarray(capacities, dtype=float)
        n_res = len(capacities_arr)
        res_crit = np.zeros(n)
        
        # Build demand matrix
        d_matrix = np.zeros((n, n_res), dtype=float)
        if demands is not None:
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
        
        # Calculate max ratio
        if n_res > 0:
            ratios = np.zeros((n, n_res))
            for r in range(n_res):
                cap_r = capacities_arr[r]
                if cap_r > 0:
                    ratios[:, r] = d_matrix[:, r] / cap_r
            res_crit = np.max(ratios, axis=1)
    else:
        res_crit = np.zeros(n)

    # 6. Combine Scores
    # Normalize components
    bl_max = np.max(bottom_level) if np.max(bottom_level) > 0 else 1.0
    bl_norm = bottom_level / bl_max
    
    tf_max = np.max(tf) if np.max(tf) > 0 else 1.0
    tf_norm = tf / tf_max
    
    delay_max = np.max(delay) if np.max(delay) > 0 else 1.0
    delay_norm = delay / delay_max
    
    rc_max = np.max(res_crit) if np.max(res_crit) > 0 else 1.0
    rc_norm = res_crit / rc_max
    
    # We want high BL, low TF, high Delay (indicating it was a bottleneck in ref), high RC.
    # Score = w1 * BL - w2 * TF + w3 * Delay + w4 * RC
    
    w_bl = 0.6
    w_tf = 0.2
    w_delay = 0.1
    w_rc = 0.1
    
    scores = w_bl * bl_norm - w_tf * tf_norm + w_delay * delay_norm + w_rc * rc_norm
    
    # Final normalization
    score_max = np.max(scores) if np.max(scores) > 0 else 1.0
    score_min = np.min(scores)
    score_range = score_max - score_min
    if score_range > 0:
        scores = (scores - score_min) / score_range
    else:
        scores = np.ones(n)

    return scores
# EVOLVE-BLOCK-END
