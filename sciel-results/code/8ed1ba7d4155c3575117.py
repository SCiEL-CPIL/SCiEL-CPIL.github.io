import numpy as np
from collections import defaultdict, deque
import time

def priority(instance):
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s limit

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    if n == 0:
        return np.array([], dtype=float)

    # Get predecessors with defensive key handling
    preds = None
    for key in ["predecessors", "preds", "predecessors_list", "pre", "pred", "precedence"]:
        if key in instance:
            preds = instance[key]
            break

    if preds is None:
        preds = [[] for _ in range(n)]

    # Normalize preds to list of lists of ints
    if isinstance(preds, np.ndarray):
        preds = preds.tolist()
        
    # Ensure preds is a list of lists
    normalized_preds = []
    for i in range(n):
        if i < len(preds):
            p_list = preds[i]
            if p_list is None:
                normalized_preds.append([])
            else:
                # Handle if p_list is a single int or array
                if isinstance(p_list, (int, np.integer)):
                    normalized_preds.append([int(p_list)])
                else:
                    normalized_preds.append([int(x) for x in p_list if x is not None])
        else:
            normalized_preds.append([])
    preds = normalized_preds

    # Build successor list
    succs = [set() for _ in range(n)]
    for j in range(n):
        for i in preds[j]:
            if 0 <= i < n and i != j:
                succs[i].add(j)

    # Topological sort
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in succs[j]:
            in_degree[i] += 1

    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_deg_copy = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                queue.append(v)

    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in set(topo_order)]
        topo_order.extend(remaining)

    # Forward pass: ES (Earliest Start)
    ES = np.zeros(n, dtype=float)
    for u in topo_order:
        for v in succs[u]:
            if ES[u] + durations[u] > ES[v]:
                ES[v] = ES[u] + durations[u]

    project_duration = np.max(ES + durations) if n > 0 else 0

    # Backward pass: LS (Latest Start)
    LS = np.full(n, project_duration, dtype=float)
    for u in reversed(topo_order):
        for v in succs[u]:
            if LS[v] - durations[u] < LS[u]:
                LS[u] = LS[v] - durations[u]

    total_float = LS - ES

    # Resource data
    demands = None
    capacities = None
    for key in ["demands", "resource_demands", "demand", "resource_demand"]:
        if key in instance:
            demands = instance[key]
            break
    for key in ["capacities", "resource_capacities", "capacity", "resource_capacity"]:
        if key in instance:
            capacities = instance[key]
            break

    num_resources = 0
    if demands is not None and capacities is not None:
        demands = np.asarray(demands, dtype=float)
        capacities = np.asarray(capacities, dtype=float)
        if demands.ndim == 1:
            demands = demands.reshape(-1, 1)
        if capacities.ndim == 0:
            capacities = capacities.reshape(1)
        if demands.shape[0] == n and capacities.shape[0] > 0:
            num_resources = capacities.shape[0]
            safe_cap = np.where(capacities > 0, capacities, 1.0)
            # Precompute resource ratios for each activity
            resource_ratios = demands / safe_cap
    else:
        resource_ratios = np.zeros((n, 1))
        num_resources = 1

    # Precompute features
    succ_count = np.array([len(s) for s in succs], dtype=float)
    
    # Strategy 1: Minimum Total Float (Standard)
    # Lower float is better, so we invert for priority (higher score = higher priority)
    # Or just use -total_float
    strat_mtf = -total_float
    
    # Strategy 2: Maximum Resource Intensity
    strat_mri = np.sum(resource_ratios, axis=1)
    
    # Strategy 3: Maximum Successors
    strat_ms = succ_count
    
    # Strategy 4: Critical Path (Longest Path from start)
    # ES is already computed, but we want distance from start? 
    # Actually, activities with high ES are "later". We want to start early ones?
    # No, CP heuristic usually prioritizes activities on the critical path.
    # A common proxy is -total_float (same as MTF).
    # Let's use a mix of ES and Duration?
    # Let's stick to distinct strategies.
    
    # Strategy 5: Combined MTF + MRI
    def normalize(x):
        if len(x) == 0:
            return x
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
        
    norm_mtf = normalize(strat_mtf)
    norm_mri = normalize(strat_mri)
    norm_ms = normalize(strat_ms)
    
    strat_comb1 = 0.5 * norm_mtf + 0.5 * norm_mri
    strat_comb2 = 0.4 * norm_mtf + 0.3 * norm_ms + 0.3 * norm_mri
    strat_comb3 = 0.7 * norm_mtf + 0.3 * norm_ms
    
    # Strategy 6: Pure Duration
    strat_dur = durations
    
    # List of candidate strategies: (name, scores)
    strategies = [
        ("MTF", strat_mtf),
        ("MRI", strat_mri),
        ("MS", strat_ms),
        ("COMB1", strat_comb1),
        ("COMB2", strat_comb2),
        ("COMB3", strat_comb3),
        ("DUR", strat_dur),
    ]

    # SSGS Simulator
    def simulate_ssgs(priority_scores):
        """
        Simulates the Serial Schedule Generation Scheme.
        Returns the makespan.
        """
        # Sort activities by priority (descending)
        # We need to respect precedence and resources.
        
        # Data structures for simulation
        # remaining_predecessors: count of unstarted predecessors
        rem_preds = np.zeros(n, dtype=int)
        for j in range(n):
            rem_preds[j] = len(preds[j])
            
        # Resource availability over time
        # We don't know the makespan in advance, so we use a list of events or a simple timeline
        # Since N is small (30), we can just keep a list of (end_time, resource_id, amount) for each resource
        # Or simpler: maintain a list of active activities and check resources at each step.
        
        # Let's use a timeline approach for resources.
        # resource_usage[r] is a list of (start, end, amount)
        # But checking overlap is O(N). With N=30, it's fine.
        
        # Better: Maintain current resource usage at the "current time" and update as we schedule.
        # However, start times are not sequential in time, they are sequential in priority order.
        # So we must find the earliest feasible start time for each activity.
        
        scheduled = [False] * n
        start_times = np.zeros(n, dtype=float)
        
        # To find earliest start time for activity j:
        # 1. Earliest time after all predecessors are finished.
        # 2. Earliest time where resources are available.
        
        # We need to track resource usage intervals.
        # resource_intervals[r] = list of [start, end, amount]
        resource_intervals = [[] for _ in range(num_resources)]
        
        # Helper to check if resource r is available at time t for duration d and demand amount a
        def is_resource_available(r, t_start, t_end, amount):
            for (s, e, a) in resource_intervals[r]:
                # Check overlap
                if t_start < e and t_end > s:
                    # Overlap exists, check if capacity is exceeded
                    # We don't store capacity here, we need to pass it or have it in scope
                    # Let's restructure: store usage and check against capacity
                    pass
            return True

        # This simple check is inefficient if we don't aggregate.
        # Let's use a different approach:
        # For each activity, we propose a start time.
        # Initial proposal: max(end time of predecessors).
        # Then, we check resources. If conflict, we find the next time slot where resources are free.
        
        # To efficiently find the next free slot, we can look at all end times of currently active tasks that conflict.
        
        def find_earliest_start(j, earliest_pred_end):
            t = earliest_pred_end
            while True:
                conflict = False
                next_conflict_end = t
                for r in range(num_resources):
                    demand = demands[j, r] if demands is not None else 0
                    if demand == 0:
                        continue
                    capacity = capacities[r]
                    # Check all intervals for resource r
                    for (s, e, a) in resource_intervals[r]:
                        if t < e and (t + durations[j]) > s:
                            # Conflict!
                            conflict = True
                            # The conflict ends at e
                            if e > next_conflict_end:
                                next_conflict_end = e
                            # Actually, we just need to jump to the end of this conflict
                            # But there might be multiple conflicts.
                            # Simplest: jump to max(end of all conflicting intervals)
                            # This might not be optimal if there's a gap, but it's a valid serial SGS.
                            # Standard SGS jumps to the end of the conflicting activity.
                            break
                if not conflict:
                    return t
                else:
                    t = next_conflict_end
                    # Safety check to avoid infinite loops if duration is 0 or something weird
                    if t > 1e9:
                        return t

        # We need to store demands and capacities in local scope for the helper
        # They are already in scope.

        # Main loop
        # Sort indices by priority
        order = np.argsort(-priority_scores)
        
        for idx in order:
            j = int(idx)
            if scheduled[j]:
                continue
            scheduled[j] = True
            
            # 1. Earliest start due to precedence
            es_pred = 0
            for i in preds[j]:
                if start_times[i] + durations[i] > es_pred:
                    es_pred = start_times[i] + durations[i]
            
            # 2. Earliest start due to resources
            t_start = find_earliest_start(j, es_pred)
            t_end = t_start + durations[j]
            
            start_times[j] = t_start
            
            # Update resource intervals
            for r in range(num_resources):
                demand = demands[j, r] if demands is not None else 0
                if demand > 0:
                    resource_intervals[r].append((t_start, t_end, demand))
                    
        makespan = np.max(start_times + durations) if n > 0 else 0
        return makespan

    # Evaluate each strategy
    best_score = float('inf')
    best_priority = None
    
    for name, scores in strategies:
        if time.time() - start_time > DEADLINE:
            # Fallback to a known good heuristic if timeout
            return strat_mtf
            
        # We need to handle the case where demands/capacities might be None in the simulator
        # The simulator uses `demands` and `capacities` from the outer scope.
        # If they are None, we set num_resources=1 and demands=0, so it works.
        
        try:
            makespan = simulate_ssgs(scores)
            if makespan < best_score:
                best_score = makespan
                best_priority = scores
        except Exception:
            continue

    if best_priority is None:
        # Fallback
        return strat_mtf

    return best_priority
