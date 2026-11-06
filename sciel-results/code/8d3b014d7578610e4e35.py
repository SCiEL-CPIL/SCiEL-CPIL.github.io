import numpy as np
import time as _time_module

_START_TIME = _time_module.perf_counter()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill


def _soft_min(values, temperature=1.0):
    """Compute soft-min using log-sum-exp trick for numerical stability."""
    if len(values) == 0:
        return 0.0
    v = np.array(values, dtype=np.float64)
    if len(v) == 1:
        return float(v[0])
    neg_v = -v
    max_neg = np.max(neg_v)
    shifted = neg_v - max_neg
    exp_vals = np.exp(shifted / temperature)
    sum_exp = np.sum(exp_vals)
    if sum_exp < 1e-300:
        return float(v[0])
    log_sum_exp = max_neg + temperature * np.log(sum_exp)
    soft_min_val = -log_sum_exp
    return float(soft_min_val)


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling (makespan minimization).
    
    Critical path slack + machine contention avoidance:
    - Compute critical path slack: (remaining_work - time) / min_duration
    - Estimate machine contention as total load on candidate machines
    - Select operation with highest slack-to-contention ratio
    - This prioritizes urgent operations on less congested machines
    """
    if not ready_operations:
        return 0

    if _time_module.perf_counter() > _DEADLINE:
        return 0

    n_ops = len(ready_operations)
    
    # First pass: compute machine queue depths and total available time
    machine_queue_depths = {}
    machine_total_time = {}
    
    soft_min_temp = 0.5
    
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue
        
        durations = list(machine_options.values())
        if not durations:
            continue
        
        # Compute soft-min duration for this operation
        sm_duration = _soft_min(durations, temperature=soft_min_temp)
        if sm_duration <= 0:
            sm_duration = 1e-9
        
        # Compute weights for distributing load across machines
        durations_arr = np.array(durations, dtype=np.float64)
        neg_durs = -durations_arr
        max_neg = np.max(neg_durs)
        shifted = neg_durs - max_neg
        weights = np.exp(shifted / soft_min_temp)
        weight_sum = np.sum(weights)
        if weight_sum < 1e-300:
            weights = np.ones(len(durations_arr))
            weight_sum = len(durations_arr)
        else:
            weights = weights / weight_sum
        
        # Distribute the operation's processing time across machines
        for j, (m, w) in enumerate(zip(machine_options.keys(), weights)):
            if m not in machine_queue_depths:
                machine_queue_depths[m] = 0.0
                machine_total_time[m] = 0.0
            machine_queue_depths[m] += sm_duration * w
            machine_total_time[m] += 1.0  # count each operation as contributing 1 unit of demand

    # Compute average queue depth for normalization
    if machine_queue_depths:
        avg_queue_depth = sum(machine_queue_depths.values()) / len(machine_queue_depths)
    else:
        avg_queue_depth = 1.0

    best_index = 0
    best_score = -float('inf')
    best_secondary = float('inf')

    epsilon = 1e-9

    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            # Fallback: no machine options
            effective_load = 1.0
            flexibility = 0.0
            sm_duration = 1.0
        else:
            durations = list(machine_options.values())
            if not durations:
                effective_load = 1.0
                flexibility = 0.0
                sm_duration = 1.0
            else:
                # Compute soft-min duration
                sm_duration = _soft_min(durations, temperature=soft_min_temp)
                if sm_duration <= 0:
                    sm_duration = 1e-9
                
                # Compute weights for machine preference
                durations_arr = np.array(durations, dtype=np.float64)
                neg_durs = -durations_arr
                max_neg = np.max(neg_durs)
                shifted = neg_durs - max_neg
                weights = np.exp(shifted / soft_min_temp)
                weight_sum = np.sum(weights)
                if weight_sum < 1e-300:
                    weights = np.ones(len(durations_arr))
                    weight_sum = len(durations_arr)
                else:
                    weights = weights / weight_sum
                
                # Effective load: weighted average of queue depths on candidate machines
                loads_arr = np.array([machine_queue_depths.get(m, 0.0) for m in machine_options.keys()], dtype=np.float64)
                effective_load = float(np.dot(weights, loads_arr))
                
                # Also consider the minimum queue depth (best case)
                min_queue_depth = float(np.min(loads_arr)) if len(loads_arr) > 0 else 0.0
                
                # Blend: use 70% weighted average + 30% minimum for robustness
                effective_load = 0.7 * effective_load + 0.3 * min_queue_depth
                
                # Flexibility: number of machines with near-min duration
                min_duration = float(np.min(durations_arr))
                if min_duration > 0:
                    threshold = min_duration * 1.2
                    flexible_count = sum(1 for d in durations if d <= threshold)
                    flexibility = flexible_count / len(durations)
                else:
                    flexibility = 0.0

        remaining_ops = op.get("remaining_ops", 1)
        if remaining_ops <= 0:
            remaining_ops = 1

        remaining_work = op.get("remaining_work", 0)
        if remaining_work is None:
            remaining_work = 0.0

        # Critical path slack: how much work remains relative to current time
        # If remaining_work > time, the job is behind schedule and urgent
        slack = max(remaining_work - time, 0.0)
        
        # Minimum duration for this operation (best case machine choice)
        if machine_options:
            min_dur = min(machine_options.values())
            if min_dur <= 0:
                min_dur = 1e-9
        else:
            min_dur = 1.0
        
        # Machine contention: ratio of queue depth to total available time
        # Higher contention = more operations competing for the same machines
        if machine_options:
            contention_scores = []
            for m in machine_options.keys():
                qd = machine_queue_depths.get(m, 0.0)
                tt = machine_total_time.get(m, 1.0)
                if tt > 0:
                    contention_scores.append(qd / tt)
            if contention_scores:
                avg_contention = np.mean(contention_scores)
            else:
                avg_contention = 0.0
        else:
            avg_contention = 0.0
        
        # Primary score: critical path slack divided by contention
        # Higher slack-to-contention ratio = more urgent and less contested
        contention_safe = max(avg_contention, 0.01)
        slack_contention_ratio = slack / (min_dur * contention_safe)
        
        # Urgency boost: jobs with more remaining ops are more critical
        urgency_factor = 1.0 + 0.15 * (remaining_ops - 1)
        
        # Flexibility bonus: more flexible operations can be rescheduled more easily
        flexibility_bonus = 1.0 + 0.1 * flexibility
        
        # Combined score
        score = slack_contention_ratio * urgency_factor * flexibility_bonus

        # Secondary: prefer lower remaining_work (job closer to completion)
        secondary = remaining_work

        if score > best_score + epsilon:
            best_score = score
            best_secondary = secondary
            best_index = i
        elif abs(score - best_score) <= epsilon:
            if secondary < best_secondary:
                best_secondary = secondary
                best_index = i

    return int(best_index)
