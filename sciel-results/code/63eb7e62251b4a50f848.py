"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque


def priority(instance):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Parse instance structure
    # Expecting: durations, demands (dict of resource->list or 2D array), capacities, predecessors
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    # Build adjacency list for predecessors
    pred_list = [list() for _ in range(n)]
    if predecessors is not None:
        if isinstance(predecessors, dict):
            for act, preds in predecessors.items():
                act_idx = int(act)
                for p in preds:
                    pred_list[act_idx].append(int(p))
        elif isinstance(predecessors, (list, tuple, np.ndarray)):
            for i, preds in enumerate(predecessors):
                if preds is not None:
                    if isinstance(preds, (list, tuple, np.ndarray)):
                        pred_list[i] = [int(p) for p in preds]
                    elif isinstance(preds, (int, np.integer)):
                        pred_list[i] = [int(preds)]
    
    # Build demands structure
    # demands could be: list of dicts, 2D array, or dict of resource->list
    n_resources = 0
    demand_matrix = None
    cap_vector = None
    
    if demands is not None:
        if isinstance(demands, (list, tuple, np.ndarray)):
            d_arr = np.asarray(demands)
            if d_arr.ndim == 2:
                demand_matrix = d_arr.astype(float)
                n_resources = d_arr.shape[1]
            elif d_arr.ndim == 1:
                # Assume single resource
                demand_matrix = d_arr.reshape(-1, 1).astype(float)
                n_resources = 1
        elif isinstance(demands, dict):
            # dict of resource -> list
            res_keys = list(demands.keys())
            n_resources = len(res_keys)
            demand_matrix = np.zeros((n, n_resources))
            for j, rk in enumerate(res_keys):
                val = demands[rk]
                if isinstance(val, (list, tuple, np.ndarray)):
                    demand_matrix[:, j] = np.asarray(val, dtype=float)
                else:
                    demand_matrix[:, j] = float(val)
    
    if capacities is not None:
        if isinstance(capacities, (list, tuple, np.ndarray)):
            cap_vector = np.asarray(capacities, dtype=float)
            if cap_vector.ndim == 0:
                cap_vector = np.full(n_resources, float(cap_vector)) if n_resources > 0 else np.array([1.0])
        elif isinstance(capacities, dict):
            cap_vector = np.zeros(n_resources)
            for rk, cv in capacities.items():
                for j, res_key in enumerate([rk] * n_resources) if n_resources > 0 else []:
                    pass
            # Simpler: if dict, try to match keys
            if n_resources > 0:
                # Assume keys match resource indices or are single
                if len(capacities) == 1:
                    # single resource
                    cap_vector = np.full(n_resources, float(list(capacities.values())[0]))
                else:
                    # Try to map
                    for j in range(n_resources):
                        if j in capacities:
                            cap_vector[j] = float(capacities[j])
                        elif str(j) in capacities:
                            cap_vector[j] = float(capacities[str(j)])
                        else:
                            cap_vector[j] = float(list(capacities.values())[j]) if j < len(capacities) else 1.0
    
    if demand_matrix is None:
        # No resource constraints
        demand_matrix = np.zeros((n, 0))
        cap_vector = np.array([])
        n_resources = 0
    
    if cap_vector is None or len(cap_vector) != n_resources:
        cap_vector = np.ones(n_resources)
    
    # Compute total float (backward pass)
    # First, compute earliest start times (forward pass)
    # Using topological order
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred_list[i]:
            in_degree[i] += 1
    
    # Topological sort (Kahn's algorithm)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    temp_in_deg = in_degree.copy()
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        # Find successors
        for j in range(n):
            if node in pred_list[j]:
                temp_in_deg[j] -= 1
                if temp_in_deg[j] == 0:
                    queue.append(j)
    
    # If cycle detected (shouldn't happen in valid RCPSP), handle gracefully
    if len(topo_order) < n:
        # Add remaining nodes
        for i in range(n):
            if i not in topo_order:
                topo_order.append(i)
    
    # Forward pass: earliest start
    earliest_start = np.zeros(n)
    for node in topo_order:
        if pred_list[node]:
            earliest_start[node] = max(earliest_start[p] + durations[p] for p in pred_list[node])
    
    # Backward pass: latest start
    # First, find which activities have no successors
    has_successor = np.zeros(n, dtype=bool)
    for i in range(n):
        for p in pred_list[i]:
            has_successor[p] = True
    
    # Project end time (makespan without resource constraints)
    project_end = max(earliest_start[i] + durations[i] for i in range(n)) if n > 0 else 0
    
    latest_finish = np.full(n, project_end)
    # For activities with no successors, latest finish = project_end
    # For others, latest finish = min(earliest start of successors)
    for node in reversed(topo_order):
        # Find successors of node
        successors = [j for j in range(n) if node in pred_list[j]]
        if successors:
            latest_finish[node] = min(earliest_start[s] for s in successors)
        else:
            latest_finish[node] = project_end
    
    latest_start = latest_finish - durations
    total_float = latest_start - earliest_start
    total_float = np.maximum(total_float, 0)
    
    # Compute resource utilization/criticality
    # Simple heuristic: resources with high demand relative to capacity are more critical
    if n_resources > 0:
        # For each resource, compute total demand and average utilization
        res_criticality = np.zeros(n_resources)
        for j in range(n_resources):
            total_demand = np.sum(demand_matrix[:, j])
            if cap_vector[j] > 0:
                res_criticality[j] = total_demand / (cap_vector[j] * project_end) if project_end > 0 else 0
            else:
                res_criticality[j] = 1.0 if np.any(demand_matrix[:, j] > 0) else 0
        
        # Activity resource criticality: weighted sum of resource criticalities
        activity_res_crit = np.zeros(n)
        for i in range(n):
            activity_res_crit[i] = np.sum(demand_matrix[i] * res_criticality)
    else:
        activity_res_crit = np.zeros(n)
    
    # Compute number of successors (immediacy)
    num_successors = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred_list[i]:
            num_successors[p] += 1
    
    # Combine heuristics into a base priority
    # Higher priority = scheduled earlier
    # Components:
    # 1. Longest remaining path (duration + sum of successor durations in critical path)
    # 2. Low total float (more critical)
    # 3. High resource criticality
    # 4. Many successors (branching)
    
    # Compute longest path from each activity to end (including its own duration)
    longest_path = np.zeros(n)
    for node in reversed(topo_order):
        successors = [j for j in range(n) if node in pred_list[j]]
        if successors:
            longest_path[node] = durations[node] + max(longest_path[s] for s in successors)
        else:
            longest_path[node] = durations[node]
    
    # Normalize components
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.zeros_like(arr)
    
    norm_duration = normalize(durations)
    norm_longest_path = normalize(longest_path)
    norm_float_inv = 1.0 - normalize(total_float)  # Low float = high priority
    norm_res_crit = normalize(activity_res_crit)
    norm_succ = normalize(num_successors.astype(float))
    
    # Base priority: weighted combination
    # Weights: longest path is most important, then duration, float, resource crit, successors
    base_priority = (
        0.35 * norm_longest_path +
        0.25 * norm_duration +
        0.20 * norm_float_inv +
        0.15 * norm_res_crit +
        0.05 * norm_succ
    )
    
    # Check time
    if time.time() > deadline - 10:
        return base_priority
    
    # Lookahead simulation for initially eligible activities
    # Only simulate for a few top-priority candidates to save time
    initial_eligible = [i for i in range(n) if len(pred_list[i]) == 0]
    
    if not initial_eligible:
        return base_priority
    
    # Sort initial eligible by base priority (descending)
    initial_eligible_sorted = sorted(initial_eligible, key=lambda x: -base_priority[x])
    
    # Simulate only top K candidates
    K = min(5, len(initial_eligible_sorted))
    top_candidates = initial_eligible_sorted[:K]
    
    # Define simulate function
    def simulate_schedule(priority_order):
        """Simulate serial schedule generation with given priority order. Returns makespan."""
        if time.time() > deadline - 5:
            return np.inf
        
        # Copy state
        remaining_predecessors = [list(pred) for pred in pred_list]
        eligible = set(initial_eligible)
        time_now = 0.0
        remaining_activities = set(range(n))
        resource_available = cap_vector.copy()
        scheduled = []
        
        # Priority ranking: lower index = higher priority
        priority_rank = {act: rank for rank, act in enumerate(priority_order)}
        
        while eligible or time_now < project_end:
            if not eligible:
                # Advance time to next event
                if not remaining_activities:
                    break
                # Find minimum completion time of scheduled activities
                # This is a simplified simulation - for full accuracy we need event-based
                # For speed, we'll just break and estimate
                break
            
            # Pick highest priority eligible
            best_act = min(eligible, key=lambda a: priority_rank.get(a, n))
            
            # Check if resources are available
            can_schedule = True
            for j in range(n_resources):
                if demand_matrix[best_act, j] > resource_available[j]:
                    can_schedule = False
                    break
            
            if not can_schedule:
                # Can't schedule best activity, try next
                # Find any eligible activity that can be scheduled
                found = False
                for act in sorted(eligible, key=lambda a: priority_rank.get(a, n)):
                    sched = True
                    for j in range(n_resources):
                        if demand_matrix[act, j] > resource_available[j]:
                            sched = False
                            break
                    if sched:
                        best_act = act
                        found = True
                        break
                if not found:
                    # No activity can be scheduled, advance time
                    # Find earliest completion time
                    # This is a simplified approach - in reality we'd need to track completion times
                    # For now, just break
                    break
            
            # Schedule best_act
            eligible.remove(best_act)
            remaining_activities.remove(best_act)
            scheduled.append((best_act, time_now, time_now + durations[best_act]))
            
            # Consume resources
            for j in range(n_resources):
                resource_available[j] -= demand_matrix[best_act, j]
            
            # Update eligible set: activities whose predecessors are all scheduled
            for act in list(remaining_activities):
                if act not in eligible and act not in [s[0] for s in scheduled]:
                    all_pred_done = True
                    for p in pred_list[act]:
                        if p not in [s[0] for s in scheduled]:
                            all_pred_done = False
                            break
                    if all_pred_done:
                        eligible.add(act)
            
            # Note: This simulation doesn't properly handle resource release over time
            # It's a simplified approximation. For a more accurate simulation, we'd need
            # to track completion events and release resources.
        
        # Compute makespan
        if scheduled:
            makespan = max(end for _, _, end in scheduled)
        else:
            makespan = project_end
        
        return makespan
    
    # Simulate for top candidates
    lookahead_scores = {}
    for act in top_candidates:
        if time.time() > deadline - 5:
            break
        
        # Create priority order: act first, then others by base priority
        others = [a for a in range(n) if a != act]
        others_sorted = sorted(others, key=lambda x: -base_priority[x])
        order = [act] + others_sorted
        
        makespan = simulate_schedule(order)
        lookahead_scores[act] = makespan
    
    # If we have lookahead scores, adjust priority for top candidates
    if lookahead_scores:
        # Lower makespan = better, so priority = -makespan
        # Combine with base priority
        for act, mk in lookahead_scores.items():
            if np.isfinite(mk):
                # Scale makespan to comparable range
                # Lower makespan should give higher priority
                # Use inverse: priority_boost = 1 / (mk + 1)
                priority_boost = 1.0 / (mk + 1.0)
                # Normalize boost
                base_priority[act] = 0.7 * base_priority[act] + 0.3 * priority_boost
    
    return base_priority
# EVOLVE-BLOCK-END
