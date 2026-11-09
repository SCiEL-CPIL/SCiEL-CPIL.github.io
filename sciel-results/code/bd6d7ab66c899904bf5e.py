"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def dispatch(ready_operations, time):
    """
    Dispatch rule for Open-Shop Scheduling.
    Strategy: Longest Processing Time (LPT) on the fastest available machine.
    1. For each ready operation, identify the machine with the shortest processing time.
    2. Select the operation that has the longest processing time on its assigned (fastest) machine.
    3. Return the index of the operation and the assigned machine.
    """
    if not ready_operations:
        return 0, 0  # Fallback, though ready_operations should not be empty if called

    best_index = -1
    best_machine = -1
    max_duration = -1.0

    # Safety margin check (though this loop is extremely fast)
    start_time = time.time()
    
    for i, op in enumerate(ready_operations):
        # Check deadline (unlikely to trigger for small N)
        if time.time() - start_time > 140:
            # Return best found so far
            if best_index != -1:
                return (best_index, best_machine)
            else:
                return (0, list(ready_operations[0]["machine_options"].keys())[0])

        machine_options = op["machine_options"]
        
        # Find the machine with the minimum duration for this operation
        min_duration = float('inf')
        assigned_machine = None
        
        for machine, duration in machine_options.items():
            if duration < min_duration:
                min_duration = duration
                assigned_machine = machine
        
        # Update best operation based on LPT (longest duration on assigned machine)
        if min_duration > max_duration:
            max_duration = min_duration
            best_index = i
            best_machine = assigned_machine

    # If no operation was found (should not happen), return first
    if best_index == -1:
        first_op = ready_operations[0]
        best_machine = list(first_op["machine_options"].keys())[0]
        return (0, best_machine)

    return (best_index, best_machine)
# EVOLVE-BLOCK-END
