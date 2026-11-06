"""Combined dispatch rule for job-shop scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time_module
import math

_START_TIME = _time_module.perf_counter()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill


def _soft_min(values, beta=2.0):
    """
    Soft-min via log-sum-exp for smooth surrogate of min().
    As beta -> inf, approaches true min. Larger beta = sharper (closer to hard min).
    Uses a numerically stable formulation.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    
    # Use log-sum-exp trick for numerical stability
    min_val = min(values)
    shifted = [v - min_val for v in values]
    
    # Soft-min: min_val - 1/beta * log(sum(exp(-beta * (v - min_val))))
    # This is a smooth approximation where beta controls sharpness
    exp_sum = 0.0
    for s in shifted:
        if -beta * s > 500:  # avoid overflow
            exp_sum += math.exp(500)
        else:
            exp_sum += math.exp(-beta * s)
    
    if exp_sum <= 0:
        return min_val
    
    return min_val - (1.0 / beta) * math.log(exp_sum)


def dispatch(ready_operations, time):
    """
    Dispatch rule for Job-Shop Scheduling.
    
    Combines the critical path slack concept (remaining_work / min_duration)
    with machine-contention avoidance via flexibility ratio and SPT bonus.
    
    Primary: critical path slack = remaining_work / soft_min_duration
    Secondary: SPT bonus = 1 / soft_min_duration
    Tertiary: flexibility (prefer operations with more machine options / shorter min)
    Quaternary: remaining work and urgency
    """
    if not ready_operations:
        return 0
    
    # Deadline guard: if we're getting close to the limit, use fast fallback
    if _time_module.perf_counter() > _DEADLINE:
        return int(np.argmax([op.get("remaining_work", 0) for op in ready_operations]))
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Compute metrics for each ready operation
    min_durations = []
    total_durations = []
    n_machines_list = []
    remaining_works = []
    remaining_ops_list = []
    
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        if not machine_options:
            min_dur = 1.0
            total_dur = 1.0
            n_machines = 1
        else:
            durations = list(machine_options.values())
            n_machines = len(durations)
            min_dur = min(durations)
            total_dur = sum(durations)
            
            # Use soft-min with sharper beta for better discrimination
            # Beta=4.0 gives a tighter approximation to hard min while remaining smooth
            if n_machines > 1:
                soft_min_val = _soft_min(durations, beta=4.0)
                # Use soft-min directly; it smoothly interpolates between min and mean
                min_dur = soft_min_val
        
        if min_dur <= 0:
            min_dur = 1e-9
        if total_dur <= 0:
            total_dur = 1e-9
        
        min_durations.append(min_dur)
        total_durations.append(total_dur)
        n_machines_list.append(n_machines)
        
        rem_work = op.get("remaining_work", 0)
        if rem_work <= 0:
            rem_work = min_dur
        remaining_works.append(rem_work)
        
        rem_ops = op.get("remaining_ops", 1)
        if rem_ops <= 0:
            rem_ops = 1
        remaining_ops_list.append(rem_ops)
    
    # Convert to numpy arrays
    min_durations = np.array(min_durations, dtype=np.float64)
    total_durations = np.array(total_durations, dtype=np.float64)
    n_machines_arr = np.array(n_machines_list, dtype=np.float64)
    remaining_works = np.array(remaining_works, dtype=np.float64)
    remaining_ops_arr = np.array(remaining_ops_list, dtype=np.float64)
    
    # Primary score: critical path slack = remaining_work / min_duration
    # This captures urgency: jobs with lots of remaining work but short current op
    # should be prioritized to keep the critical path moving
    with np.errstate(divide='ignore', invalid='ignore'):
        primary_score = remaining_works / min_durations
        primary_score = np.where(np.isfinite(primary_score), primary_score, 0.0)
    
    # Secondary: SPT bonus - pure shortest processing time preference
    with np.errstate(divide='ignore', invalid='ignore'):
        spt_score = 1.0 / min_durations
        spt_score = np.where(np.isfinite(spt_score), spt_score, 0.0)
    
    # Tertiary: flexibility ratio - min_dur / total_dur
    # Lower ratio means more flexible (can pick shorter machine)
    # Higher score for more flexible operations
    with np.errstate(divide='ignore', invalid='ignore'):
        flexibility_ratio = min_durations / total_durations
        flexibility_ratio = np.where(np.isfinite(flexibility_ratio), flexibility_ratio, 1.0)
    flexibility_score = 1.0 - flexibility_ratio  # Higher = more flexible
    
    # Quaternary: number of machine options (more options = less contention)
    num_options_score = n_machines_arr
    
    # Quinary: remaining work - prioritize jobs with more total work
    work_score = remaining_works
    
    # Senary: fewer remaining ops means job is closer to completion
    ops_score = -remaining_ops_arr
    
    # Normalize scores to [0, 1] range for better weighting
    def normalize(arr):
        if len(arr) == 0:
            return arr
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    primary_norm = normalize(primary_score)
    spt_norm = normalize(spt_score)
    flexibility_norm = normalize(flexibility_score)
    options_norm = normalize(num_options_score)
    work_norm = normalize(work_score)
    ops_norm = normalize(ops_score)
    
    # Composite score with tuned weights
    # Primary (critical path slack) is the dominant signal
    # SPT provides additional weight to very short operations
    # Flexibility avoids machine contention
    # Remaining work and ops provide secondary effects
    # Increased primary weight to emphasize critical path; reduced noise terms
    scores = (
        4.5 * primary_norm +          # Primary: critical path slack (dominant)
        1.0 * spt_norm +              # Secondary: pure SPT
        0.8 * flexibility_norm +      # Machine-contention avoidance
        0.1 * options_norm +          # More machine options = less contention
        0.2 * work_norm +             # Tertiary: remaining work
        0.1 * ops_norm                # Quaternary: fewer remaining ops
    )
    
    # Select the operation with the highest score
    best_idx = int(np.argmax(scores))
    
    return int(best_idx)
# EVOLVE-BLOCK-END
