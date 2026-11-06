"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque
import random

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
    
    # 4. MCTS-based Dynamic Priority Refinement
    # We simulate the schedule using the current priority, then use MCTS to evaluate
    # how much each activity's placement affects the final makespan.
    
    current_priority = initial_priority.copy()
    
    # MCTS Parameters
    # With 30 activities, we can afford a moderate number of simulations.
    # We will run a few rounds of "Lookahead" simulations.
    
    max_mcts_rounds = 5
    simulations_per_round = 20
    
    for r in range(max_mcts_rounds):
        if time.time() > deadline - 10.0:
            break
            
        # 1. Simulate a baseline schedule with current priorities
        start_times, end_times, scheduled_order = simulate_schedule(
            durations, pred_list, succ_list, demand_matrix, cap_vector, current_priority, n
        )
        
        if scheduled_order is None or len(scheduled_order) < n:
            break
            
        # 2. Identify "Critical" activities in this schedule
        # Activities on the critical path of the generated schedule are high impact.
        # We can approximate this by checking which activities, if delayed, would delay the project end.
        # Or simpler: Run MCTS from the state *before* each activity was scheduled?
        # That's too expensive (30 * 20 sims).
        
        # Alternative MCTS Strategy:
        # Instead of full MCTS, use a "Regret" heuristic derived from random sampling.
        # For each activity i, estimate its "criticality" by seeing how much the makespan
        # varies when we randomly shuffle the order of activities with similar priority.
        
        # A more direct MCTS approach for Priority Calculation:
        # We treat the scheduling process as a tree.
        # State: (scheduled_set, current_time, resource_usage)
        # Action: Schedule next activity.
        # We want to find which action leads to the shortest path to the end.
        
        # Since we only need to return a priority score, we can use the results of
        # random rollouts to estimate the "value" of scheduling each eligible activity now.
        
        # Let's define a simpler dynamic score:
        # "Resource Contention Intensity"
        # For each activity, how much does it block others?
        
        # Let's stick to the MCTS-inspired "Lookahead" but simplified:
        # We will perform random restarts of the scheduling process starting from the
        # state where only the first K activities are fixed (or none).
        
        # Actually, a very effective and fast heuristic that mimics MCTS value estimation
        # is to compute the "Expected Delay" caused by each activity.
        
        # Let's compute a "Bottleneck Score" based on the generated schedule.
        # For each activity i, look at its successors.
        # If a successor s was delayed relative to its earliest start, attribute some of that
        # delay to i if i was running or just finished.
        
        dynamic_score = np.zeros(n)
        
        # Calculate actual delay for each node
        actual_delay = start_times - earliest_start
        actual_delay = np.maximum(actual_delay, 0)
        
        # Propagate delay backwards?
        # If node i is a predecessor of s, and s is delayed, i might be the cause.
        # Specifically, if s could not start because resources were held by i (or i's predecessors).
        
        # Let's use a "Blocking" metric:
        # For each pair (i, s) where i is pred of s:
        # If start_times[s] > end_times[i], the gap is not due to i.
        # If start_times[s] == end_times[i] (or close), i is on the critical path of the schedule.
        
        # Better: Use the "Slack" in the generated schedule.
        # Calculate the "Free Slack" for each activity in the generated schedule.
        # Free Slack = min(start_times[s] - end_times[i]) for all successors s.
        # If Free Slack is 0, the activity is critical in this schedule.
        
        free_slack = np.full(n, float('inf'))
        for i in range(n):
            if succ_list[i]:
                min_succ_start = min(start_times[s] for s in succ_list[i])
                free_slack[i] = min_succ_start - end_times[i]
            else:
                free_slack[i] = project_end - end_times[i] # Distance to end
        
        # Normalize free slack (lower slack = higher priority)
        norm_slack = 1.0 - normalize(free_slack)
        
        # Combine with static priority
        # The MCTS intuition is that we should prioritize actions that lead to lower cost states.
        # Here, low free slack means the activity is "tight" in the current solution.
        # We want to keep these tight activities early to avoid breaking the critical path.
        
        alpha = 0.5 + 0.1 * r # Increase reliance on dynamic score over time
        beta = 1.0 - alpha
        
        new_priority = alpha * norm_slack + beta * initial_priority
        
        # Optional: Add a small random noise to explore?
        # No, we want stable priorities for the final return.
        
        current_priority = new_priority
        
    # Final return
    if len(current_priority) != n:
        return initial_priority
        
    return current_priority


def simulate_schedule(durations, pred_list, succ_list, demand_matrix, cap_vector, priority, n):
    """
    Serial Schedule Generation Scheme.
    Returns: start_times, end_times, scheduled_order
    """
    n_resources = demand_matrix.shape[1]
    
    scheduled = [False] * n
    start_times = np.zeros(n)
    end_times = np.zeros(n)
    scheduled_order = []
    
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
        es_pred = 0
        for p in pred_list[best]:
            es_pred = max(es_pred, end_times[p])
            
        es_res = es_pred
        
        t = es_pred
        max_attempts = n + 1
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            feasible = True
            for r in range(n_resources):
                demand_needed = demand_matrix[best][r]
                if demand_needed == 0:
                    continue
                usage = 0.0
                for i in range(n):
                    if i == best or not scheduled[i]:
                        continue
                    if start_times[i] < t and end_times[i] > t:
                        usage += demand_matrix[i][r]
                    
                if usage + demand_needed > cap_vector[r] + 1e-9:
                    feasible = False
                    min_end = float('inf')
                    for i in range(n):
                        if i == best or not scheduled[i]:
                            continue
                        if start_times[i] < t and end_times[i] > t and demand_matrix[i][r] > 0:
                            if end_times[i] < min_end:
                                min_end = end_times[i]
                    if min_end == float('inf'):
                        t = t + 1.0
                    else:
                        t = min_end
                    break
            
            if feasible:
                break
                
        start_times[best] = t
        end_times[best] = t + durations[best]
        scheduled[best] = True
        scheduled_order.append(best)
        remaining.remove(best)
        
    return start_times, end_times, scheduled_order
# EVOLVE-BLOCK-END
