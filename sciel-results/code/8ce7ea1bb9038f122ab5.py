"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    best_score = -1
    best_index = 0
    best_machine = None

    # Estimate cumulative load for each machine based on current ready operations
    machine_loads = {}
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        for machine, duration in machine_options.items():
            if machine not in machine_loads:
                machine_loads[machine] = 0
            machine_loads[machine] += duration

    for i, op in enumerate(ready_operations):
        # Get remaining work for the job (proxy for critical path length)
        job_remaining_work = op.get("remaining_work", 0)
        
        # Get machine options
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue
            
        # Find the machine with minimum estimated cumulative load
        # Among machines with similar load, prefer shorter duration
        best_machine_for_op = None
        best_machine_score = float('inf')
        
        for machine, duration in machine_options.items():
            load = machine_loads.get(machine, 0)
            # Score: prioritize machines with lower load, then shorter duration
            # Use a composite score: load * (1 + small factor for duration)
            machine_score = load + 0.1 * duration
            if machine_score < best_machine_score:
                best_machine_score = machine_score
                best_machine_for_op = machine
        
        if best_machine_for_op is None:
            continue
            
        min_duration = machine_options[best_machine_for_op]
        
        # Avoid division by zero
        if min_duration == 0:
            min_duration = 1e-9

        # Calculate score: Ratio of job criticality to estimated machine load
        # This prioritizes long jobs that can run on less congested machines
        load = machine_loads.get(best_machine_for_op, 0)
        if load == 0:
            load = 1e-9

        score = job_remaining_work / load
        
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_machine_for_op

    return (best_index, best_machine)
# EVOLVE-BLOCK-END
