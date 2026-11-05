"""Improved dispatch rule for Flexible Job-Shop Scheduling."""

import math
import time as _time

# Module-level cache for machine availability estimates
# Key: (time_bucket, machine_id) -> estimated available time
# We use a coarse time bucket to avoid infinite growth
_mach_avail_cache = {}
_cache_last_time = [0.0]

def _estimate_mach_avail(machine_id, current_time, remaining_work_total, num_machines):
    """
    Estimate when a machine will be available.
    Uses a simple heuristic: assume machines are shared equally among remaining work.
    """
    # Coarse bucket to limit cache size
    bucket = int(current_time * 10) / 10.0
    key = (bucket, machine_id)
    
    if key in _mach_avail_cache:
        return _mach_avail_cache[key]
    
    # Simple estimate: if we have R total remaining work across all machines,
    # each machine handles R/num_machines on average.
    # The machine is available at approximately: current_time + (R/num_machines - already_assigned)
    # Since we don't track already_assigned, use a simple proportional estimate.
    # A machine with more remaining work assigned to it is busier.
    # Here we use: available ≈ current_time + remaining_work_total / num_machines * 0.5
    # The 0.5 factor is a heuristic that the machine has already done some work.
    
    est = current_time + (remaining_work_total / max(num_machines, 1)) * 0.5
    
    _mach_avail_cache[key] = est
    
    # Prune cache if too large
    if len(_mach_avail_cache) > 10000:
        _mach_avail_cache.clear()
    
    return est

def dispatch(ready_operations, time):
    """
    Picks the ready operation to start next and the best machine for it.
    
    Strategy:
    1. For each ready operation, use a soft-min surrogate to estimate effective processing time,
       which smoothly accounts for all machine options rather than just the best one.
    2. Estimate machine availability using remaining work as a proxy for load.
    3. Calculate a priority score combining urgency (remaining work), efficiency,
       and machine load balancing to avoid bottlenecks.
    4. Select the operation with the highest score.
    5. Return the index of the operation and the ID of the best machine.
    """
    if not ready_operations:
        return 0
    
    # Deadline guard: track elapsed time
    start_time = _time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Temperature parameter for soft-min: lower = closer to hard min
    # Tighter temperature for sharper machine selection
    temp = 1.0
    
    # Estimate total remaining work for machine availability heuristic
    total_remaining_work = sum(op.get('remaining_work', 0) for op in ready_operations)
    num_machines = 5  # From task description
    
    best_index = 0
    best_score = -float('inf')
    best_machine = None
    best_min_dur = float('inf')
    
    for i, op in enumerate(ready_operations):
        # Check deadline
        if _time.time() > deadline:
            break
        
        remaining_work = op.get('remaining_work', 0)
        machine_options = op.get('machine_options', {})
        
        if not machine_options:
            continue

        # Soft-min: -temp * log(sum(exp(-duration/temp)))
        # This gives a smooth approximation to min(duration) that considers all options
        log_sum = 0.0
        min_duration = float('inf')
        min_machine = None
        for machine_id, duration in machine_options.items():
            if duration < min_duration:
                min_duration = duration
                min_machine = machine_id
            # Use log-sum-exp for numerical stability
            log_sum += math.exp(-duration / temp)
        
        if min_machine is None:
            continue
        
        # Soft-min effective duration
        soft_min_dur = -temp * math.log(log_sum)
        
        # Estimate when the best machine will be available
        mach_avail = _estimate_mach_avail(min_machine, time, total_remaining_work, num_machines)
        
        # Effective start time: max(current time, machine availability)
        effective_start = max(time, mach_avail)
        
        # Total time this operation will occupy the machine (from now)
        effective_completion = effective_start + soft_min_dur
        
        # Machine load balancing: penalize if the best machine is likely busy
        remaining_ops = op.get('remaining_ops', 1)
        load_factor = 1.0 + 0.15 * remaining_ops  # Penalty for jobs with many remaining ops
        
        # Score: High remaining work, low effective completion time, balanced load
        epsilon = 1e-6
        urgency = remaining_work / (effective_completion + epsilon)
        efficiency = 1.0 / (effective_completion + epsilon)
        
        # Combine: urgency is primary, efficiency as secondary
        score = urgency * 0.65 + efficiency * 0.35
        score /= load_factor
        
        # Tie-breaking: Prefer shorter effective completion, then fewer remaining ops
        if score > best_score + 1e-12:
            best_score = score
            best_index = i
            best_machine = min_machine
            best_min_dur = effective_completion
        elif abs(score - best_score) <= 1e-12:
            # Prefer shorter effective completion
            if effective_completion < best_min_dur - 1e-9:
                best_index = i
                best_machine = min_machine
                best_min_dur = effective_completion
            elif abs(effective_completion - best_min_dur) <= 1e-9:
                # Prefer fewer remaining operations (less future contention)
                if remaining_ops < op.get('remaining_ops', float('inf')):
                    best_index = i
                    best_machine = min_machine
                    best_min_dur = effective_completion

    return (best_index, best_machine)
