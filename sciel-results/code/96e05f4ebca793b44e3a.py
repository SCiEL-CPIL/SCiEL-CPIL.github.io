import numpy as np
import time

def priority(instance):
    t_start = time.time()
    DEADLINE = 145.0
    SAFETY_MARGIN = 5.0

    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    if "capacabilities" in instance:
        capacities = np.asarray(instance["capacabilities"], dtype=float)
    else:
        capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]

    n = len(durations)
    n_res = len(capacities)

    if n == 0:
        return []

    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)

    # Topological order (Kahn's)
    in_degree = [len(predecessors[i]) for i in range(n)]
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    temp_in_degree = in_degree[:]
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            temp_in_degree[v] -= 1
            if temp_in_degree[v] == 0:
                queue.append(v)
    
    if len(topo_order) < n:
        # Cycle or invalid, return durations as fallback
        return durations.copy()

    # Forward pass: Head (ES + duration)
    head = np.zeros(n)
    for i in topo_order:
        if predecessors[i]:
            head[i] = durations[i] + max(head[p] for p in predecessors[i])
        else:
            head[i] = durations[i]

    # Backward pass: Tail (duration + longest path to end)
    tail = np.zeros(n)
    for i in reversed(topo_order):
        if not successors[i]:
            tail[i] = durations[i]
        else:
            tail[i] = durations[i] + max(tail[s] for s in successors[i])

    project_end = np.max(tail) if n > 0 else 0.0

    # Total Float
    # slack = project_end - tail[i] - (head[i] - durations[i])
    # ES[i] = head[i] - durations[i]
    # LS[i] = project_end - tail[i]
    # Slack = LS - ES = project_end - tail[i] - head[i] + durations[i]
    total_float = project_end - tail - head + durations

    # Resource metrics
    if n_res > 0:
        safe_caps = np.maximum(capacities, 1e-9)
        demand_ratios = demands / safe_caps
        max_demand_ratio = np.max(demand_ratios, axis=1)
        sum_demand_ratio = np.sum(demand_ratios, axis=1)
    else:
        max_demand_ratio = np.zeros(n)
        sum_demand_ratio = np.zeros(n)

    # Normalizations
    max_float = np.max(total_float) if np.max(total_float) > 0 else 1.0
    max_tail = np.max(tail) if np.max(tail) > 0 else 1.0
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    max_ratio = np.max(max_demand_ratio) if np.max(max_demand_ratio) > 0 else 1.0
    max_sum_ratio = np.max(sum_demand_ratio) if np.max(sum_demand_ratio) > 0 else 1.0
    
    # Successor count
    succ_count = np.array([len(s) for s in successors], dtype=float)
    max_succ = np.max(succ_count) if np.max(succ_count) > 0 else 1.0
    succ_norm = succ_count / max_succ

    # Predecessor count
    pred_count = np.array([len(p) for p in predecessors], dtype=float)
    max_pred = np.max(pred_count) if np.max(pred_count) > 0 else 1.0
    pred_norm = pred_count / max_pred

    # Check deadline
    if time.time() - t_start > DEADLINE - SAFETY_MARGIN:
        # Fallback: CPM + Resource
        return (
            0.5 * (max_float - total_float) / max_float +
            0.3 * tail / max_tail +
            0.2 * max_demand_ratio / max_ratio
        )

    # Conflict Graph & Eigenvector Centrality (from current program)
    precedes = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for p in predecessors[i]:
            precedes[p][i] = True
            
    for k in range(n):
        for i in range(n):
            if precedes[i][k]:
                precedes[i] |= precedes[k]

    conflict = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            if precedes[i][j] or precedes[j][i]:
                continue
            if n_res > 0:
                for r in range(n_res):
                    if demands[i][r] + demands[j][r] > capacities[r]:
                        conflict[i][j] = 1.0
                        conflict[j][i] = 1.0
                        break

    try:
        vec = np.ones(n) / n
        for _ in range(50):
            new_vec = conflict @ vec
            norm = np.linalg.norm(new_vec)
            if norm < 1e-15:
                break
            new_vec /= norm
            if np.linalg.norm(new_vec - vec) < 1e-10:
                break
            vec = new_vec
        max_centrality = np.max(vec) if np.max(vec) > 0 else 1.0
        eig_centrality = vec / max_centrality
    except Exception:
        eig_centrality = np.zeros(n)

    # Check deadline
    if time.time() - t_start > DEADLINE - SAFETY_MARGIN:
        return (
            0.4 * (max_float - total_float) / max_float +
            0.3 * tail / max_tail +
            0.2 * eig_centrality +
            0.1 * max_demand_ratio / max_ratio
        )

    # Improved Serial Schedule Simulation
    # Key improvement: Efficient resource checking using event-driven updates
    # and a better baseline rule that incorporates Tail and Successor Count.
    
    def simulate_serial_schedule(rule_func):
        scheduled = [False] * n
        start_times = np.zeros(n)
        end_times = np.zeros(n)
        
        # Resource usage over time is complex to track globally efficiently without a segment tree
        # for n=30, simple iteration is fine, but we can optimize the search loop.
        
        # Pre-compute eligible list updates? No, eligibility changes as we schedule.
        
        for _ in range(n):
            if time.time() - t_start > DEADLINE - SAFETY_MARGIN:
                break
            
            eligible = []
            for i in range(n):
                if not scheduled[i]:
                    if all(scheduled[p] for p in predecessors[i]):
                        eligible.append(i)
            
            if not eligible:
                break
            
            best_activity = max(eligible, key=lambda i: rule_func(i, start_times, end_times, scheduled))
            
            earliest_start = 0.0
            for p in predecessors[best_activity]:
                earliest_start = max(earliest_start, end_times[p])
            
            # Find earliest feasible start time
            t = earliest_start
            # Max search time: project_end + max_dur is usually enough, but let's be safe
            max_search_time = project_end + max_dur * 5
            
            while t < max_search_time:
                can_start = True
                for r in range(n_res):
                    usage = 0.0
                    # Optimization: only check scheduled activities that overlap [t, t+durations[best_activity])
                    for j in range(n):
                        if scheduled[j] and start_times[j] < t + durations[best_activity] and end_times[j] > t:
                            usage += demands[j][r]
                    if usage + demands[best_activity][r] > capacities[r]:
                        can_start = False
                        break
                
                if can_start:
                    break
                
                # Find next event time (end of some active activity)
                next_event = None
                for j in range(n):
                    if scheduled[j] and start_times[j] < t < end_times[j]:
                        if next_event is None or end_times[j] < next_event:
                            next_event = end_times[j]
                
                if next_event is not None and next_event > t:
                    t = next_event
                else:
                    t += 1.0
            
            start_times[best_activity] = t
            end_times[best_activity] = t + durations[best_activity]
            scheduled[best_activity] = True

        return start_times, end_times

    # Baseline Rule: 
    # Combine Criticality (Tail/Float) with Structural Importance (Eig) and Resource Pressure.
    # Inspiration 1 emphasized CPM Rank (Tail) heavily. Inspiration 2 emphasized Tail/Slack.
    # Current program used -Float.
    # Let's use a blend: High Tail, Low Float, High Eig, High Resource Pressure.
    
    def baseline_rule(i, st, et, sched):
        # Normalize components for the rule to avoid scale issues
        # Tail: higher is better
        # Float: lower is better (so -float)
        # Eig: higher is better
        # Resource: higher is better (to free up resources early)
        
        # Weights:
        # Tail is the strongest signal for makespan reduction in RCPSP heuristics.
        # Float is also strong.
        # Eig captures structural bottlenecks.
        
        return (
            1.0 * (tail[i] / max_tail) +
            0.8 * ((max_float - total_float[i]) / max_float) +
            0.5 * eig_centrality[i] +
            0.3 * (max_demand_ratio[i] / max_ratio) +
            0.2 * succ_norm[i]
        )

    sim_start, sim_end = simulate_serial_schedule(baseline_rule)

    # Check deadline
    if time.time() - t_start > DEADLINE - SAFETY_MARGIN:
        return (
            0.4 * (max_float - total_float) / max_float +
            0.3 * tail / max_tail +
            0.2 * eig_centrality +
            0.1 * max_demand_ratio / max_ratio
        )

    # Use simulated end times.
    # Activities that finish early in the baseline schedule are likely to be "easy" or "early" in the graph.
    # However, for priority, we want to schedule "hard" or "critical" activities first.
    # The simulation tells us the *actual* order that works well for the baseline.
    # If we prioritize activities that finished early in the simulation, we might replicate the good schedule.
    # Or, we can prioritize activities that had high priority in the simulation rule.
    
    # Let's stick to the idea that the simulation reveals a good ordering.
    # Priority score should be high for activities that were scheduled early in the good simulation.
    # Invert the end times: earlier end time = higher priority.
    
    max_sim_end = np.max(sim_end) if np.max(sim_end) > 0 else 1.0
    # If an activity wasn't scheduled (shouldn't happen), give it low priority
    sim_priority = 1.0 - (sim_end / max_sim_end)
    
    # Combine with static features
    # The simulation is dynamic and captures resource interactions.
    # Static features capture graph structure.
    
    w_sim = 0.60
    w_float = 0.15
    w_tail = 0.10
    w_eig = 0.05
    w_res = 0.05
    w_succ = 0.05

    priority_scores = (
        w_sim * sim_priority +
        w_float * (max_float - total_float) / max_float +
        w_tail * tail / max_tail +
        w_eig * eig_centrality +
        w_res * max_demand_ratio / max_ratio +
        w_succ * succ_norm
    )

    return np.maximum(priority_scores, 0.0)
