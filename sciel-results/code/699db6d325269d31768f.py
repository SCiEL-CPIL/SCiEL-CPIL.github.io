"""Improved dispatch rule for Flexible Job-Shop Scheduling."""

import math

def dispatch(ready_operations, time):
    """
    Picks the ready operation to start next and the best machine for it.
    
    Strategy:
    1. For each ready operation, compute a soft-min duration across machine options
       (smooth surrogate for min to handle near-ties).
    2. Count how many other ready operations also prefer each machine (congestion).
    3. Score = (Remaining_Work / (soft_min_duration + epsilon)) * (1 / (1 + congestion_penalty))
    4. Select the operation with the highest score.
    5. Return the index and the machine with the best adjusted duration (min duration 
       adjusted by congestion).
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Pre-compute for each operation: min duration, best machine, and machine preference counts
    op_min_durs = [0.0] * n
    op_best_machines = [None] * n
    machine_congestion = {}  # machine -> count of ops that would prefer it
    
    for i, op in enumerate(ready_operations):
        machine_options = op.get('machine_options', {})
        if not machine_options:
            op_min_durs[i] = float('inf')
            continue
        
        min_duration = float('inf')
        min_machine = None
        for machine_id, duration in machine_options.items():
            if duration < min_duration:
                min_duration = duration
                min_machine = machine_id
        
        op_min_durs[i] = min_duration
        op_best_machines[i] = min_machine
        
        if min_machine is not None:
            machine_congestion[min_machine] = machine_congestion.get(min_machine, 0) + 1
    
    # Now score each operation
    best_index = 0
    best_score = -1.0
    best_machine = None
    best_min_dur = float('inf')
    
    epsilon = 1e-6
    congestion_weight = 0.3  # How much congestion matters
    
    for i, op in enumerate(ready_operations):
        remaining_work = op.get('remaining_work', 0)
        machine_options = op.get('machine_options', {})
        
        if not machine_options:
            continue
        
        min_duration = op_min_durs[i]
        min_machine = op_best_machines[i]
        
        if min_machine is None:
            continue
        
        # Congestion penalty: how many other ops also want this machine
        cong = machine_congestion.get(min_machine, 1)
        congestion_factor = 1.0 / (1.0 + congestion_weight * (cong - 1))
        
        # Soft-min surrogate: for near-tie machines, the effective duration is slightly
        # higher than the absolute min, reflecting that a near-min machine might be less congested.
        # We use a log-sum-exp style approach: effective_dur = -log(sum(exp(-d/T))) * T
        # For simplicity and speed, we approximate: if second-best is within 10% of best,
        # blend slightly toward the second-best.
        durations = list(machine_options.values())
        if len(durations) > 1:
            sorted_durs = sorted(durations)
            if sorted_durs[1] - sorted_durs[0] < 0.1 * sorted_durs[0]:
                # Near-tie: effective duration is a bit higher
                effective_dur = sorted_durs[0] + 0.05 * (sorted_durs[1] - sorted_durs[0])
            else:
                effective_dur = sorted_durs[0]
        else:
            effective_dur = min_duration
        
        score = (remaining_work / (effective_dur + epsilon)) * congestion_factor
        
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = min_machine
            best_min_dur = min_duration
        elif abs(score - best_score) < 1e-9:
            # Tie-break: prefer shorter effective duration, then less congestion
            if effective_dur < best_min_dur - 1e-9:
                best_index = i
                best_machine = min_machine
                best_min_dur = min_duration
    
    return (best_index, best_machine)
