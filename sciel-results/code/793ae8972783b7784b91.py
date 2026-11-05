"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Track start time to ensure we stay well within the 120s/150s limits
# Although SPT is O(N), the guard is cheap and safe per directive.
_start_time = _time.time()
_DEADLINE = 140.0  # 10s margin before 150s kill

def dispatch(ready_operations, time):
    """
    Dispatch rule: Shortest Processing Time (SPT).
    
    Strategy: Select the ready operation with the shortest processing time 
    among its available machine options. This tends to clear the queue quickly,
    potentially reducing waiting times for subsequent operations and lowering
    makespan compared to Longest Processing Time (LPT) or static work-based rules.
    """
    # Deadline guard: if we are running out of time, return the first valid index
    # to avoid timeouts. In practice, SPT is fast enough that this won't trigger
    # unless the instance is pathological or the system is overloaded.
    if _time.time() - _start_time > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    min_duration = float('inf')
    best_idx = 0
    
    # Iterate through ready operations to find the one with the shortest processing time
    # We assume machine_options is a dict {machine: duration}
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            # Fallback if no options (shouldn't happen in valid job-shop)
            continue
        
        # Get the minimum duration among available machines for this operation
        # SPT typically considers the processing time on the assigned machine.
        # Since we don't know the assignment yet, we look at the minimum possible
        # duration to estimate "shortness". Alternatively, one could argue for
        # the duration on the *assigned* machine if known, but the interface
        # suggests we pick the operation, implying the simulator assigns the machine
        # or we pick the machine too. The prompt says "return index ... (optionally (index, machine))".
        # The seed returned just index. We will stick to index to be safe, 
        # but use the min duration as the metric for "shortest".
        
        # If the simulator allows us to pick the machine, picking the fastest machine
        # for the shortest operation is the most aggressive SPT.
        # If the simulator assigns the machine based on other factors, we still
        # prioritize the op that *can* be done fastest.
        
        op_min_duration = min(machine_options.values())
        
        if op_min_duration < min_duration:
            min_duration = op_min_duration
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
