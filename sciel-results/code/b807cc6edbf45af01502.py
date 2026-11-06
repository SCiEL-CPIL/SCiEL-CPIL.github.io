"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Track machine loads across dispatch calls for load balancing.
# Since the simulation is event-driven and we're called repeatedly,
# we maintain a running estimate of total work assigned to each machine.
_MACHINE_LOAD = {}
_CALL_COUNT = [0]

def dispatch(ready_operations, time):
    """
    Dispatch rule: earliest-completion-time with critical-path weighting.
    
    Structural approach:
    1. For each ready operation, evaluate ALL eligible machines (not just shortest).
    2. For each (op, machine) pair, compute estimated completion time =
       current_time + estimated_queue_wait_on_machine + duration.
    3. Weight by criticality: operations with more remaining_work and 
       remaining_ops (longer critical path) get priority.
    4. Select the (op, machine) pair that minimizes a combined objective:
       completion_time * (1 + criticality_weight)
    
    This is a structural change from static normalized scoring to a
    dynamic load-aware earliest-completion heuristic.
    """
    if not ready_operations:
        return 0

    _CALL_COUNT[0] += 1
    n = len(ready_operations)
    
    # For each operation, find the best (machine, duration) considering load
    best_score = float('inf')
    best_idx = 0
    best_machine = None
    
    for i in range(n):
        op = ready_operations[i]
        machine_options = op["machine_options"]
        if not machine_options:
            continue
        
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Criticality: higher remaining work and more remaining ops = more urgent
        # Use a multiplicative factor to amplify criticality
        criticality = 1.0 + 0.1 * remaining_work + 0.05 * remaining_ops
        
        for m, dur in machine_options.items():
            load = _MACHINE_LOAD.get(m, 0.0)
            # Estimated completion: current time + queue wait (approx by load) + duration
            # We use load as a proxy for how busy the machine is
            est_completion = time + load + dur
            
            # Objective: minimize completion time, weighted by criticality
            # Higher criticality means we care more about finishing this op soon
            score = est_completion * criticality
            
            if score < best_score:
                best_score = score
                best_idx = i
                best_machine = m
                best_dur = dur
    
    # Update machine load for the selected machine
    if best_machine is not None and best_machine in ready_operations[best_idx]["machine_options"]:
        _MACHINE_LOAD[best_machine] = _MACHINE_LOAD.get(best_machine, 0.0) + best_dur
    
    return int(best_idx)
# EVOLVE-BLOCK-END
