"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # First pass: gather all machine options and durations to estimate machine loads
    # We'll track the minimum duration for each machine across all ready operations
    # to estimate congestion
    
    machine_min_durations = {}
    all_ops_info = []
    
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue
            
        job_remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        
        # Track minimum duration per machine for load estimation
        for machine, duration in machine_options.items():
            if machine not in machine_min_durations:
                machine_min_durations[machine] = duration
            else:
                machine_min_durations[machine] = min(machine_min_durations[machine], duration)
        
        all_ops_info.append({
            'index': i,
            'job_remaining_work': job_remaining_work,
            'remaining_ops': remaining_ops,
            'machine_options': machine_options
        })
    
    if not all_ops_info:
        return 0

    # Second pass: score each operation based on criticality vs machine congestion
    best_score = -1
    best_index = 0
    best_machine = None

    for info in all_ops_info:
        i = info['index']
        job_remaining_work = info['job_remaining_work']
        remaining_ops = info['remaining_ops']
        machine_options = info['machine_options']
        
        # Estimate the critical path length more accurately
        # Use remaining_work as primary, but also consider remaining_ops
        # A better estimate of critical path: remaining_work is the sum of min durations
        # of remaining operations, which is a lower bound on makespan contribution
        
        # For each eligible machine, calculate a congestion-aware score
        best_op_score = -1
        best_op_machine = None
        
        for machine, duration in machine_options.items():
            # Estimated load on this machine: minimum duration among all ready ops
            # that can use this machine (proxy for how busy it is)
            machine_load = machine_min_durations.get(machine, duration)
            
            # Total estimated time if we use this machine
            # We want to avoid machines that are already heavily loaded
            # Score: prioritize operations that have high criticality (remaining work)
            # and low machine congestion
            
            # The ratio: job criticality / (operation duration + machine congestion factor)
            # Higher job remaining work means more urgent
            # Lower machine load means less congestion
            
            # Use a weighted combination:
            # Primary: remaining_work / (duration * (1 + machine_load/duration))
            # This penalizes operations on congested machines
            
            congestion_factor = 1.0 + (machine_load / max(duration, 1e-9))
            score = job_remaining_work / (duration * congestion_factor)
            
            # Also factor in remaining operations count as a tiebreaker
            # Jobs with more remaining operations might be more critical
            if remaining_ops > 1:
                score *= (1.0 + 0.1 * (remaining_ops - 1))
            
            if score > best_op_score:
                best_op_score = score
                best_op_machine = machine
        
        if best_op_machine is None:
            continue
            
        if best_op_score > best_score:
            best_score = best_op_score
            best_index = i
            best_machine = best_op_machine

    return (best_index, best_machine)
# EVOLVE-BLOCK-END
