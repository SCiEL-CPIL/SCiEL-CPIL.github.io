import numpy as np
import time

def priority(instance):
    t_start = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0

    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # (n, n_res)
    capacities = np.asarray(instance["capacities"], dtype=float)  # (n_res,)
    predecessors = instance["predecessors"]  # list of lists
    
    n = len(durations)
    n_res = len(capacities)
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Forward pass: ES, EF
    es = np.zeros(n)
    ef = np.zeros(n)
    for i in range(n):
        for p in predecessors[i]:
            es[i] = max(es[i], ef[p])
        ef[i] = es[i] + durations[i]
    
    project_end = np.max(ef)
    
    # Backward pass: LS, LF
    ls = np.zeros(n)
    lf = np.zeros(n)
    for i in range(n-1, -1, -1):
        if not successors[i]:
            lf[i] = project_end
        else:
            lf[i] = min(ls[s] for s in successors[i])
        ls[i] = lf[i] - durations[i]
    
    # Total float
    total_float = ls - es
    
    # Tail: distance from EF to project end
    tail = project_end - ef
    
    # Resource demand intensity: total demand relative to capacity
    if n_res > 0:
        demand_ratios = demands / np.maximum(capacities, 1e-9)  # (n, n_res)
        max_demand_ratio = np.max(demand_ratios, axis=1)  # (n,)
        # Sum of demand ratios (total resource pressure)
        sum_demand_ratio = np.sum(demand_ratios, axis=1)  # (n,)
    else:
        max_demand_ratio = np.zeros(n)
        sum_demand_ratio = np.zeros(n)
    
    # Normalize features
    max_float = np.max(total_float) if np.max(total_float) > 0 else 1.0
    max_tail = np.max(tail) if np.max(tail) > 0 else 1.0
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    max_ratio = np.max(max_demand_ratio) if np.max(max_demand_ratio) > 0 else 1.0
    max_sum_ratio = np.max(sum_demand_ratio) if np.max(sum_demand_ratio) > 0 else 1.0
    
    # Check deadline before computing final score
    if time.time() - t_start > DEADLINE - SAFETY_MARGIN:
        # Return a simple fallback based on criticality
        return (max_float - total_float) / max_float
    
    # Dynamic Look-Ahead Simulation:
    # Run a fast serial schedule simulation with a baseline rule to estimate
    # each activity's actual completion time under resource constraints.
    # Use these estimated completion times (inverted) as priority scores.
    
    def simulate_serial_schedule(rule_func):
        """Run serial schedule generation with a given priority rule.
        Returns estimated completion times for each activity."""
        scheduled = [False] * n
        start_times = np.zeros(n)
        end_times = np.zeros(n)
        current_time = 0.0
        
        # Track resource usage over time
        # We'll use a simple event-based approach
        events = []  # (time, activity, is_start)
        
        for _ in range(n):
            if time.time() - t_start > DEADLINE - SAFETY_MARGIN:
                break
            
            # Find eligible activities (all predecessors completed)
            eligible = []
            for i in range(n):
                if not scheduled[i]:
                    # Check if all predecessors are scheduled
                    all_preds_done = all(scheduled[p] for p in predecessors[i])
                    if all_preds_done:
                        eligible.append(i)
            
            if not eligible:
                break
            
            # Apply the rule to pick the highest priority eligible activity
            best_activity = max(eligible, key=lambda i: rule_func(i, start_times, end_times, scheduled))
            
            # Determine start time considering resource constraints
            # Simple approach: find earliest time when resources are available
            earliest_start = current_time
            for p in predecessors[best_activity]:
                earliest_start = max(earliest_start, end_times[p])
            
            # Check resource availability from earliest_start
            # We need to find a time slot where all resources have enough capacity
            t = earliest_start
            max_search_time = project_end + max_dur * 2
            while t < max_search_time:
                # Check if resources are available at time t
                can_start = True
                for r in range(n_res):
                    # Count current usage of resource r at time t
                    usage = 0.0
                    for j in range(n):
                        if scheduled[j] and start_times[j] <= t < end_times[j]:
                            usage += demands[j][r]
                    if usage + demands[best_activity][r] > capacities[r]:
                        can_start = False
                        break
                
                if can_start:
                    break
                
                # Find next event time (when some activity finishes)
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
            current_time = t
        
        return start_times, end_times
    
    # Baseline rule: prioritize by longest duration, then by least float
    def baseline_rule(i, st, et, sched):
        # Higher score = higher priority
        # Use negative float (less float = more critical) and duration
        return -total_float[i] + durations[i]
    
    # Run simulation with baseline rule
    sim_start, sim_end = simulate_serial_schedule(baseline_rule)
    
    # Check deadline
    if time.time() - t_start > DEADLINE - SAFETY_MARGIN:
        # Fallback to static CPM-based priority
        priority_scores = (
            0.45 * (max_float - total_float) / max_float +
            0.30 * tail / max_tail +
            0.10 * durations / max_dur +
            0.08 * max_demand_ratio / max_ratio +
            0.07 * sum_demand_ratio / max_sum_ratio
        )
        return np.maximum(priority_scores, 0.0)
    
    # Use simulated end times as priority (earlier completion = higher priority)
    # Invert: activities that complete earlier in simulation get higher priority
    max_sim_end = np.max(sim_end) if np.max(sim_end) > 0 else 1.0
    sim_priority = 1.0 - sim_end / max_sim_end
    
    # Combine with CPM signals for robustness
    # Weight the dynamic simulation results heavily, but blend with static features
    w_sim = 0.60
    w_float = 0.20
    w_tail = 0.10
    w_res_max = 0.05
    w_res_sum = 0.05
    
    priority_scores = (
        w_sim * sim_priority +
        w_float * (max_float - total_float) / max_float +
        w_tail * tail / max_tail +
        w_res_max * max_demand_ratio / max_ratio +
        w_res_sum * sum_demand_ratio / max_sum_ratio
    )
    
    # Ensure non-negative scores
    priority_scores = np.maximum(priority_scores, 0.0)
    
    return priority_scores
