"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next using a Shortest Processing Time (SPT) Heuristic.
    
    This approach differs from urgency-based heuristics by prioritizing the
    completion of the current operation to free up machine resources as soon
    as possible. It selects the operation with the shortest processing time
    on its optimal machine. Ties are broken by higher remaining work (urgency)
    and lower machine flexibility (constraint).
    
    Rationale: In Job-Shop Scheduling, blocking is a major source of delay.
    By prioritizing short jobs, we reduce the time a machine is occupied,
    allowing subsequent operations from other jobs to start earlier, which
    can significantly reduce the makespan in dense schedules.
    """
    if not ready_operations:
        return 0
    
    best_key = None
    best_index = 0
    
    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            # If no machine options, this op is invalid/stuck, skip or penalize heavily
            continue
        
        num_machines = len(machine_options)
        durations = list(machine_options.values())
        min_duration = min(durations)
        
        # Primary Key: Shortest Processing Time (lower is better)
        # Secondary Key: Remaining Work (higher is better)
        # Tertiary Key: Constraint (lower num_machines is better, i.e., higher 1/num)
        
        # We construct a tuple to be minimized:
        # (min_duration, -remaining_work, num_machines)
        # Lower min_duration is better.
        # Higher remaining_work is better (so we negate it for minimization).
        # Lower num_machines is better (more constrained, so we keep it as is for minimization? 
        # Actually, if we want to prioritize constrained, we want to pick smaller num_machines.
        # So (min_duration, -remaining_work, num_machines) works if we minimize.)
        
        key = (min_duration, -remaining_work, num_machines)
        
        if best_key is None or key < best_key:
            best_key = key
            best_index = i
            
    return best_index
# EVOLVE-BLOCK-END
