"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque

def priority(instance):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # 1. Parse Instance
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    # Build adjacency list for predecessors and successors
    pred_list = [list() for _ in range(n)]
    succ_list = [list() for _ in range(n)]
    
    if predecessors is not None:
        if isinstance(predecessors, dict):
            for act, preds in predecessors.items():
                act_idx = int(act)
                for p in preds:
                    p_idx = int(p)
                    if act_idx not in succ_list[p_idx]:
                        succ_list[p_idx].append(act_idx)
                    if p_idx not in pred_list[act_idx]:
                        pred_list[act_idx].append(p_idx)
        elif isinstance(predecessors, (list, tuple, np.ndarray)):
            for i, preds in enumerate(predecessors):
                if preds is not None:
                    if isinstance(preds, (list, tuple, np.ndarray)):
                        for p in preds:
                            p_idx = int(p)
                            if i not in succ_list[p_idx]:
                                succ_list[p_idx].append(i)
                            if p_idx not in pred_list[i]:
                                pred_list[i].append(p_idx)
                    elif isinstance(preds, (int, np.integer)):
                        p_idx = int(preds)
                        if i not in succ_list[p_idx]:
                            succ_list[p_idx].append(i)
                        if p_idx not in pred_list[i]:
                            pred_list[i].append(p_idx)
    
    # Build demands structure
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
                demand_matrix = d_arr.reshape(-1, 1).astype(float)
                n_resources = 1
        elif isinstance(demands, dict):
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
            if n_resources > 0:
                if len(capacities) == 1:
                    cap_vector = np.full(n_resources, float(list(capacities.values())[0]))
                else:
                    for j in range(n_resources):
                        if j in capacities:
                            cap_vector[j] = float(capacities[j])
                        elif str(j) in capacities:
                            cap_vector[j] = float(capacities[str(j)])
                        else:
                            cap_vector[j] = float(list(capacities.values())[j]) if j < len(capacities) else 1.0
    
    if demand_matrix is None:
        demand_matrix = np.zeros((n, 0))
        cap_vector = np.array([])
        n_resources = 0
    
    if cap_vector is None or len(cap_vector) != n_resources:
        cap_vector = np.ones(n_resources)

    # 2. Topological Sort & Initial Path Calculations
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        in_degree[i] = len(pred_list[i])
        
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    temp_in_deg = in_degree.copy()
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for j in succ_list[node]:
            temp_in_deg[j] -= 1
            if temp_in_deg[j] == 0:
                queue.append(j)
                
    if len(topo_order) < n:
        for i in range(n):
            if i not in topo_order:
                topo_order.append(i)

    # Forward Pass: Earliest Start
    earliest_start = np.zeros(n)
    for node in topo_order:
        if pred_list[node]:
            earliest_start[node] = max(earliest_start[p] + durations[p] for p in pred_list[node])
            
    project_end = max(earliest_start[i] + durations[i] for i in range(n)) if n > 0 else 0
    
    # Backward Pass: Latest Start
    latest_finish = np.full(n, project_end)
    for node in reversed(topo_order):
        if succ_list[node]:
            latest_finish[node] = min(earliest_start[s] for s in succ_list[node])
        else:
            latest_finish[node] = project_end
            
    latest_start = latest_finish - durations
    total_float = latest_start - earliest_start
    total_float = np.maximum(total_float, 0)
    
    # Longest Path from node to end
    longest_path = np.zeros(n)
    for node in reversed(topo_order):
        if succ_list[node]:
            longest_path[node] = durations[node] + max(longest_path[s] for s in succ_list[node])
        else:
            longest_path[node] = durations[node]

    # 3. Initial Priority based on Static Heuristics
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.zeros_like(arr)
        
    # Initial weights: emphasize critical path and low float
    initial_priority = (
        0.5 * normalize(longest_path) +
        0.3 * (1.0 - normalize(total_float)) +
        0.2 * normalize(durations)
    )
    
    # 4. Iterative Refinement with Dynamic Resource Contention
    # We simulate the schedule with the current priority, then compute a "blocking score"
    # for each activity based on how much it delays its successors.
    
    current_priority = initial_priority.copy()
    
    # Number of iterations depends on time budget
    max_iters = 10
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
            
        # Simulate Serial Schedule Generation
        # Returns: start_times, end_times, and a list of activities in scheduled order
        start_times, end_times, scheduled_order = simulate_schedule(
            durations, pred_list, succ_list, demand_matrix, cap_vector, current_priority, n
        )
        
        if scheduled_order is None or len(scheduled_order) < n:
            break
            
        # Calculate Dynamic Blocking Score
        # For each activity i, look at its successors.
        # If a successor s had to wait longer than its earliest start (due to resources),
        # attribute that delay partially to i if i was running or recently finished.
        # Simplified: Delay of s = start_times[s] - earliest_start[s]
        # We want to penalize activities that are "bottlenecks".
        
        dynamic_score = np.zeros(n)
        
        # Map activity to its index in scheduled order for quick lookup? 
        # Or just use start_times directly.
        
        # For each activity i:
        # It contributes to the delay of its successors if those successors were delayed.
        # Let's define "impact" of i as the sum of delays of its immediate successors.
        # But we also want to consider transitive impact.
        # A simple proxy: For each successor s of i, if s was delayed, add that delay to i's score.
        
        # However, a more robust "contention" measure:
        # How much resource was locked by i that prevented s from starting?
        # This is hard to compute exactly without detailed logs.
        
        # Alternative Dynamic Signal: 
        # "Regret" or "Delay Propagation".
        # Let's compute the actual delay caused by resource constraints for each node.
        # delay[i] = start_times[i] - earliest_start[i]
        
        actual_delay = start_times - earliest_start
        actual_delay = np.maximum(actual_delay, 0)
        
        # Now, propagate this "delay burden" backwards.
        # If a node is delayed, it likely delayed its predecessors less, but it might be delaying its successors.
        # Actually, we want to prioritize activities that *cause* delays.
        # If activity i is running, and activity s (successor) is waiting, i is causing the delay.
        
        # Let's use a "Delay Propagation" metric:
        # For each node i, score = sum of (delay[s] for s in successors) * (duration[i] / (duration[i] + duration[s]))
        # This weights the delay by how much i "blocks" s.
        
        # Simpler: Just sum the delays of immediate successors.
        # If I have many successors that are delayed, I am a bottleneck.
        
        for i in range(n):
            if succ_list[i]:
                s_delay_sum = sum(actual_delay[s] for s in succ_list[i])
                dynamic_score[i] = s_delay_sum
            else:
                # Sinks don't delay anything, but if they are on the critical path, they matter.
                # Keep their static priority influence?
                # Let's just use their own delay as a proxy for "being late".
                dynamic_score[i] = actual_delay[i]
        
        # Combine Static and Dynamic
        # Normalize dynamic score
        norm_dynamic = normalize(dynamic_score)
        
        # Blend: The dynamic score tells us what actually happened in the last simulation.
        # We want to fix the bottlenecks identified.
        # So, high dynamic score = high priority (to try and resolve the bottleneck earlier).
        
        # We also keep some static signal to avoid oscillation.
        alpha = 0.6  # Weight for dynamic
        beta = 1.0 - alpha
        
        new_priority = alpha * norm_dynamic + beta * initial_priority
        
        # Check for convergence or improvement
        # If the priority order hasn't changed much, stop.
        if it > 2:
            # Simple check: correlation between old and new
            if np.corrcoef(current_priority, new_priority)[0,1] > 0.99:
                current_priority = new_priority
                break
        
        current_priority = new_priority
        
    # Final return
    # Ensure we return a valid array
    if len(current_priority) != n:
        return initial_priority
        
    return current_priority


def simulate_schedule(durations, pred_list, succ_list, demand_matrix, cap_vector, priority, n):
    """
    Serial Schedule Generation Scheme.
    Returns: start_times, end_times, scheduled_order
    """
    n_resources = demand_matrix.shape[1]
    
    # Track resource usage over time
    # Since time is continuous/discrete, we use a list of (time, resource_delta) events or a dict.
    # For 30 activities, a simple list of active activities at any time t is fine.
    # Or, since we schedule one by one, we can just check availability at the proposed start time.
    
    # Proposed start time for each activity
    # An activity is eligible if all predecessors are finished.
    # We pick the highest priority eligible activity.
    
    scheduled = [False] * n
    start_times = np.zeros(n)
    end_times = np.zeros(n)
    scheduled_order = []
    
    # Resource usage: list of (end_time, resource_index, amount)
    # To check if resource r is available at time t for amount d:
    # Sum of amounts of all activities using r that started before t and end after t.
    
    # For efficiency with small n, we can just iterate through scheduled activities.
    
    remaining = set(range(n))
    
    while remaining:
        # Find eligible activities
        eligible = []
        for i in remaining:
            if all(scheduled[p] for p in pred_list[i]):
                eligible.append(i)
                
        if not eligible:
            # Deadlock or cycle? Break.
            break
            
        # Pick highest priority
        best = max(eligible, key=lambda x: priority[x])
        
        # Find earliest start time for best
        # 1. Predecessor constraint
        es_pred = 0
        for p in pred_list[best]:
            es_pred = max(es_pred, end_times[p])
            
        # 2. Resource constraint
        es_res = es_pred
        # We need to find the earliest time t >= es_pred such that for all resources r,
        # usage_r(t) + demand[best][r] <= cap[r]
        
        # Get all relevant end times of currently scheduled activities that overlap with es_pred
        # Actually, we just need to check at es_pred, and if it fails, jump to the next event time.
        
        # Collect all end times of scheduled activities
        # To be efficient, we can just check es_pred, then if conflict, find the min end time of conflicting activities.
        
        t = es_pred
        # Max iterations to avoid infinite loop in case of weird data
        max_attempts = n + 1
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            # Check resource availability at t
            feasible = True
            for r in range(n_resources):
                demand_needed = demand_matrix[best][r]
                if demand_needed == 0:
                    continue
                # Check usage at time t
                usage = 0.0
                for i in range(n):
                    if i == best or not scheduled[i]:
                        continue
                    # If activity i is active at time t
                    if start_times[i] < t and end_times[i] > t:
                        usage += demand_matrix[i][r]
                    # Note: If end_times[i] == t, it is not active at t (assuming [start, end) or start inclusive, end exclusive)
                    # Standard RCPSP: [start, end). So if end == t, it's free.
                    
                if usage + demand_needed > cap_vector[r] + 1e-9:
                    feasible = False
                    # Find the earliest end time among conflicting activities
                    # We need to jump to the next time a resource frees up.
                    # Find min end_time of activities i such that i is active at t and uses resource r.
                    min_end = float('inf')
                    for i in range(n):
                        if i == best or not scheduled[i]:
                            continue
                        if start_times[i] < t and end_times[i] > t and demand_matrix[i][r] > 0:
                            if end_times[i] < min_end:
                                min_end = end_times[i]
                    if min_end == float('inf'):
                        # Should not happen if feasible was False
                        t = t + 1.0
                    else:
                        t = min_end
                    break # Re-check all resources at new t
            
            if feasible:
                break
                
        start_times[best] = t
        end_times[best] = t + durations[best]
        scheduled[best] = True
        scheduled_order.append(best)
        remaining.remove(best)
        
    return start_times, end_times, scheduled_order
# EVOLVE-BLOCK-END
