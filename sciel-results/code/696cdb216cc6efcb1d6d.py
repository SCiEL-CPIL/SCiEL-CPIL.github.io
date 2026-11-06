import numpy as np
import time as _time

_start_time = _time.time()
_DEADLINE = 148.0  # Safety margin before 150s kill

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0, 0 # Fallback, shouldn't happen

    # Deadline guard: if we're running out of time, return the first operation on its first machine
    if _time.time() - _start_time > _DEADLINE:
        return 0, list(ready_operations[0]['machine_options'].keys())[0]

    # Calculate contention for each machine among ready operations
    machine_contention = {}
    for op in ready_operations:
        for m in op['machine_options']:
            machine_contention[m] = machine_contention.get(m, 0) + 1

    best_score = -float('inf')
    best_idx = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        # Constraint tightness: number of machine options (fewer = more constrained)
        num_options = len(op['machine_options'])
        
        # Use remaining_work as a measure of urgency/criticality
        remaining_work = op.get('remaining_work', 1.0)
        remaining_ops = op.get('remaining_ops', 1)
        
        # Ensure we don't divide by zero or have negative work
        if remaining_work <= 0:
            remaining_work = 1.0
        if remaining_ops <= 0:
            remaining_ops = 1
        
        # Constraint tightness factor: operations with fewer options are more critical
        # This prevents future starvation by prioritizing constrained operations
        constraint_factor = 1.0 + (5.0 - num_options) * 0.3  # Scale based on typical problem size
        
        # Critical path estimate: remaining_work / remaining_ops gives average work per op
        # Operations with high average work per remaining op are on the critical path
        avg_work_per_op = remaining_work / remaining_ops
        
        for m, duration in op['machine_options'].items():
            if duration <= 0:
                continue
            
            contention = machine_contention.get(m, 0)
            
            # Heuristic: 
            # Numerator: How much work is left (urgency) * constraint tightness * critical path factor
            # Denominator: Time it takes on this machine, penalized by how busy the machine is
            # This favors operations with high remaining work, few options, and high avg work per op
            # on fast, less busy machines.
            
            # Smooth contention penalty: use sqrt to soften the impact of high contention
            # This avoids over-penalizing machines that are moderately busy
            load_factor = 1.0 + 0.4 * np.sqrt(contention)
            effective_duration = duration * load_factor
            
            # Score combines urgency, constraint tightness, critical path factor, and machine efficiency
            # The critical path factor (avg_work_per_op) helps identify operations that will
            # determine the makespan
            score = (remaining_work * constraint_factor * np.sqrt(avg_work_per_op)) / effective_duration
            
            if score > best_score:
                best_score = score
                best_idx = i
                best_machine = m

    # Fallback if no valid option found
    if best_machine is None:
        return 0, list(ready_operations[0]['machine_options'].keys())[0]

    return best_idx, best_machine
