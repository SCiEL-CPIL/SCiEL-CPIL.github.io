"""
Rewritten program for Resource-Constrained Project Scheduling.
Uses vectorized simulation and enhanced feature set with delay feedback.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

_TIME_LIMIT = 145.0  # Safety margin before 150s kill
_START_TIME = None

def _build_graph_and_features(instance):
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    
    # Parse predecessors
    pred_lists = []
    for p in instance.get("predecessors", [None] * n):
        if p is None:
            pred_lists.append([])
        elif isinstance(p, (list, tuple, np.ndarray)):
            pred_lists.append([x for x in p if x is not None])
        else:
            pred_lists.append([p])
            
    # Build successors and in-degrees
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
    in_deg_copy = in_degree[:]
    while queue:
        u = queue.popleft()
        top_order.append(u)
        for v in successors[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                queue.append(v)
                
    if len(top_order) < n:
        # Fallback for cycles or invalid input
        top_order = list(range(n))
        
    # CPM Forward Pass (Earliest Start/Finish)
    es = np.zeros(n)
    ef = np.zeros(n)
    for u in top_order:
        for v in successors[u]:
            if es[v] < ef[u]:
                es[v] = ef[u]
        ef[u] = es[u] + durations[u]
        
    project_end = np.max(ef) if n > 0 else 0.0
    
    # CPM Backward Pass (Latest Start/Finish)
    lf = np.full(n, project_end)
    ls = np.zeros(n)
    for u in reversed(top_order):
        for v in successors[u]:
            if lf[u] > ls[v]:
                lf[u] = ls[v]
        ls[u] = lf[u] - durations[u]
        
    total_float = ls - es
    
    # Bottom Level (Successor Depth)
    bottom_level = np.zeros(n)
    for u in reversed(top_order):
        max_succ_bl = 0.0
        for v in successors[u]:
            if bottom_level[v] > max_succ_bl:
                max_succ_bl = bottom_level[v]
        bottom_level[u] = durations[u] + max_succ_bl
        
    # Resource Data
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    if demands is not None and capacities is not None:
        num_res = len(capacities)
        demand_matrix = np.zeros((n, num_res), dtype=float)
        
        if isinstance(demands, list):
            if len(demands) > 0:
                if isinstance(demands[0], dict):
                    for i, d in enumerate(demands):
                        for r, val in d.items():
                            if r < num_res:
                                demand_matrix[i, r] = val
                else:
                    dm = np.asarray(demands, dtype=float)
                    if dm.shape[0] == n:
                        if dm.ndim == 1:
                            demand_matrix[:, 0] = dm
                        else:
                            cols = min(dm.shape[1], num_res)
                            demand_matrix[:, :cols] = dm[:, :cols]
        
        capacities_arr = np.asarray(capacities, dtype=float)
        safe_caps = np.where(capacities_arr > 0, capacities_arr, 1.0)
        
        # Resource Pressure: Sum of demand/capacity ratios
        res_pressure = np.sum(demand_matrix / safe_caps, axis=1)
        
        # Successor Resource Pressure: Sum of res_pressure of successors
        succ_res_pressure = np.zeros(n)
        for u in range(n):
            for v in successors[u]:
                succ_res_pressure[u] += res_pressure[v]
                
    else:
        demand_matrix = np.zeros((n, 1), dtype=float)
        capacities_arr = np.array([1.0])
        safe_caps = np.array([1.0])
        res_pressure = np.zeros(n)
        succ_res_pressure = np.zeros(n)
        
    # --- Feature Engineering ---
    
    # 1. Normalized Total Float (Inverse: Lower float -> Higher priority)
    float_max = np.max(total_float)
    if float_max > 0:
        f_float = 1.0 - (total_float / float_max)
    else:
        f_float = np.ones(n)
        
    # 2. Bottom Level (Normalized)
    bl_max = np.max(bottom_level)
    if bl_max > 0:
        f_bl = bottom_level / bl_max
    else:
        f_bl = np.ones(n)
        
    # 3. Duration (Normalized)
    d_max = np.max(durations)
    if d_max > 0:
        f_dur = durations / d_max
    else:
        f_dur = np.ones(n)
        
    # 4. Successor Count (Normalized)
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    sc_max = np.max(succ_count)
    if sc_max > 0:
        f_sc = succ_count / sc_max
    else:
        f_sc = np.zeros(n)
        
    # 5. Resource Pressure (Normalized)
    rp_max = np.max(res_pressure)
    if rp_max > 0:
        f_rp = res_pressure / rp_max
    else:
        f_rp = np.zeros(n)
        
    # 6. Successor Resource Pressure (Normalized)
    srp_max = np.max(succ_res_pressure)
    if srp_max > 0:
        f_srp = succ_res_pressure / srp_max
    else:
        f_srp = np.zeros(n)
        
    # 7. Weighted Successor Float (Sum of floats of successors, inverted)
    ws_float = np.zeros(n)
    for u in range(n):
        for v in successors[u]:
            ws_float[u] += total_float[v]
    wf_max = np.max(ws_float)
    if wf_max > 0:
        f_wf = 1.0 - (ws_float / wf_max)
    else:
        f_wf = np.ones(n)
        
    # Base features
    base_features = np.vstack([
        f_float, 
        f_bl, 
        f_dur, 
        f_sc, 
        f_rp, 
        f_srp, 
        f_wf
    ]).T
    
    return base_features, successors, durations, demand_matrix, capacities_arr, pred_lists, es, total_float

def _simulate_schedule_fast(pred_lists, successors, durations, demand_matrix, capacities_arr, priorities, es=None):
    """
    Vectorized serial schedule generation.
    Returns makespan and delays if es is provided.
    """
    n = len(durations)
    num_res = len(capacities_arr)
    
    remaining = np.ones(n, dtype=bool)
    res_usage = np.zeros(num_res, dtype=float)
    active_tasks = [] # List of [finish_time, task_id]
    start_times = np.full(n, np.nan)
    finish_times = np.full(n, np.nan)
    
    # Pre-compute predecessor counts for quick check? 
    # Actually, checking finish_times[p] <= t_now is O(preds). 
    # With n=30, this is small.
    
    t_now = 0.0
    delays = np.zeros(n)
    
    # Initial eligible check
    while np.any(remaining):
        # Find eligible tasks: remaining AND all preds finished
        eligible_indices = []
        for i in range(n):
            if not remaining[i]:
                continue
            ready = True
            for p in pred_lists[i]:
                if np.isnan(finish_times[p]) or finish_times[p] > t_now + 1e-9:
                    ready = False
                    break
            if ready:
                eligible_indices.append(i)
                
        if not eligible_indices:
            # No eligible tasks, jump to next event
            if not active_tasks:
                break # Should not happen if graph is valid
            next_finish = min(t for t, _ in active_tasks)
            # Release resources for tasks finishing at next_finish
            for ft, tid in list(active_tasks):
                if ft <= next_finish + 1e-9:
                    active_tasks.remove([ft, tid])
                    res_usage -= demand_matrix[tid]
            t_now = next_finish
            continue
            
        # Check resource feasibility for eligible tasks
        # eligible_indices is a list of ints
        # We want to find the highest priority task that fits in resources
        # Sort eligible by priority descending
        eligible_indices.sort(key=lambda x: -priorities[x])
        
        started_any = False
        for i in eligible_indices:
            # Check if demand[i] + res_usage <= capacities
            if np.all(demand_matrix[i] + res_usage <= capacities_arr + 1e-9):
                # Start task
                res_usage += demand_matrix[i]
                start_times[i] = t_now
                finish_times[i] = t_now + durations[i]
                if es is not None:
                    delays[i] = max(0.0, t_now - es[i])
                remaining[i] = False
                active_tasks.append([finish_times[i], i])
                started_any = True
                # In serial SGS, we usually schedule ONE task at a time at a given time
                # Then re-evaluate eligibility at the same time?
                # Standard Serial SGS:
                # 1. Determine eligible set at time t.
                # 2. Select highest priority task j from eligible.
                # 3. Schedule j at t.
                # 4. Repeat.
                # So we break after starting one task.
                break
                
        if not started_any:
            # No eligible task could be scheduled due to resources
            if not active_tasks:
                break # Deadlock?
            next_finish = min(t for t, _ in active_tasks)
            for ft, tid in list(active_tasks):
                if ft <= next_finish + 1e-9:
                    active_tasks.remove([ft, tid])
                    res_usage -= demand_matrix[tid]
            t_now = next_finish
            continue
            
    max_time = np.nanmax(finish_times) if not np.all(np.isnan(finish_times)) else 0.0
    if es is not None:
        return max_time, delays
    return max_time

def priority(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Fallback: return durations as priority (longest first)
        durations = np.asarray(instance["durations"], dtype=float)
        return durations

    n = len(instance["durations"])
    if n == 0:
        return np.array([])
        
    # Build features
    base_features, successors, durations, demand_matrix, capacities_arr, pred_lists, es, total_float = _build_graph_and_features(instance)
    
    # --- Generate Delay Feedback Features ---
    # Run a few baseline schedules to get delays
    # Baseline 1: Min Float (High priority for low float)
    # Base features order: [f_float, f_bl, f_dur, f_sc, f_rp, f_srp, f_wf]
    # f_float is 1 - normalized_float. So high f_float = low float.
    # Min Float heuristic: Prioritize high f_float.
    w_min_float = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    p_min_float = base_features @ w_min_float
    _, delays_mf = _simulate_schedule_fast(pred_lists, successors, durations, demand_matrix, capacities_arr, p_min_float, es)
    
    # Baseline 2: Min Duration (High priority for long duration)
    w_min_dur = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    p_min_dur = base_features @ w_min_dur
    _, delays_md = _simulate_schedule_fast(pred_lists, successors, durations, demand_matrix, capacities_arr, p_min_dur, es)
    
    # Baseline 3: Min Bottom Level (High priority for high bottom level)
    w_min_bl = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    p_min_bl = base_features @ w_min_bl
    _, delays_mb = _simulate_schedule_fast(pred_lists, successors, durations, demand_matrix, capacities_arr, p_min_bl, es)
    
    # Average delay
    avg_delay = (delays_mf + delays_md + delays_mb) / 3.0
    ad_max = np.max(avg_delay)
    if ad_max > 1e-9:
        f_delay = avg_delay / ad_max
    else:
        f_delay = np.zeros(n)
        
    # Add delay feature to base features
    # Order: [f_float, f_bl, f_dur, f_sc, f_rp, f_srp, f_wf, f_delay]
    features = np.hstack([base_features, f_delay[:, None]])
    n_features = features.shape[1]
    
    # --- Objective Function ---
    def objective(weights):
        w_sum = np.sum(weights)
        if w_sum < 1e-9:
            w_norm = np.ones(n_features) / n_features
        else:
            w_norm = weights / w_sum
        priorities = features @ w_norm
        makespan = _simulate_schedule_fast(pred_lists, successors, durations, demand_matrix, capacities_arr, priorities)
        return makespan
    
    # --- Optimization ---
    # Initial weights: Emphasize Float, Bottom Level, Delay
    # Indices: 0:float, 1:bl, 2:dur, 3:sc, 4:rp, 5:srp, 6:wf, 7:delay
    w_current = np.array([0.30, 0.25, 0.05, 0.05, 0.10, 0.10, 0.05, 0.10])
    w_current = np.clip(w_current, 0, None)
    w_current = w_current / np.sum(w_current)
    
    best_makespan = objective(w_current)
    best_weights = w_current.copy()
    
    np.random.seed(42)
    
    # Phase 1: Random Search
    # We have ~140s. Simulation is fast now. Let's do many iterations.
    start_opt = time.time()
    max_iter_p1 = 20000
    for i in range(max_iter_p1):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        # Randomly perturb 1-2 dimensions
        n_perturb = np.random.choice([1, 2], p=[0.7, 0.3])
        dims = np.random.choice(n_features, size=n_perturb, replace=False)
        
        w_new = best_weights.copy()
        for d in dims:
            # Adaptive step size: larger early, smaller later
            progress = i / max_iter_p1
            step_size = 0.15 * (1.0 - 0.7 * progress)
            w_new[d] += np.random.uniform(-step_size, step_size)
        
        w_new = np.clip(w_new, 0, None)
        w_sum = np.sum(w_new)
        if w_sum < 1e-9:
            w_new = best_weights.copy()
        else:
            w_new = w_new / w_sum
            
        makespan = objective(w_new)
        
        if makespan < best_makespan - 1e-6:
            best_makespan = makespan
            best_weights = w_new.copy()
            
    # Phase 2: Coordinate Descent Polish
    if time.time() - _START_TIME < _TIME_LIMIT - 5:
        max_iter_p2 = 10000
        for i in range(max_iter_p2):
            if time.time() - _START_TIME > _TIME_LIMIT - 2:
                break
            d = np.random.randint(n_features)
            w_new = best_weights.copy()
            # Fine step
            w_new[d] += np.random.uniform(-0.02, 0.02)
            w_new = np.clip(w_new, 0, None)
            w_sum = np.sum(w_new)
            if w_sum < 1e-9:
                w_new = best_weights.copy()
            else:
                w_new = w_new / w_sum
                
            makespan = objective(w_new)
            if makespan < best_makespan - 1e-7:
                best_makespan = makespan
                best_weights = w_new.copy()
                
    final_priorities = features @ best_weights
    return final_priorities

# EVOLVE-BLOCK-END
