# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global state to track estimated machine loads
# We assume machine IDs are consistent.
_machine_load = {}
_start_time = None

def _init_state():
    global _start_time
    if _start_time is None:
        _start_time = _time.time()

def _get_machine_load(m):
    return _machine_load.get(m, 0.0)

def _update_machine_load(m, duration):
    if m not in _machine_load:
        _machine_load[m] = 0.0
    _machine_load[m] += duration

def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Returns:
        int: index of the ready operation to start
        or (int, machine): index and chosen machine if multiple options
    """
    _init_state()
    
    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Deadline guard
    if _time.time() - _start_time > 140:
        # Fast fallback: return first
        return 0

    best_idx = -1
    best_machine = None
    best_score = float('inf')
    
    # We will try to return (index, machine) to control assignment
    # Heuristic:
    # 1. For each op, find the best machine (min load + duration)
    # 2. Score the op based on:
    #    - Estimated finish time on best machine (lower is better)
    #    - Remaining work (higher is better, to unlock downstream ops early? or lower to finish job?)
    #    - Let's try: Minimize (est_finish) but penalize if the job has very little work left?
    #    Actually, "Most Work Remaining" (MWR) is often good for makespan as it keeps pipeline full.
    #    "Shortest Processing Time" (SPT) is good for average flow time.
    #    For makespan, MWR + Load Balancing is a strong candidate.
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        remaining_work = op["remaining_work"]
        
        # Find best machine for this op based on current load estimates
        min_est_finish = float('inf')
        chosen_machine = None
        for m, dur in opts.items():
            load = _get_machine_load(m)
            est_finish = load + dur
            if est_finish < min_est_finish:
                min_est_finish = est_finish
                chosen_machine = m
        
        # Score calculation
        # We want to minimize makespan.
        # If we pick an op with high remaining work, we might finish the job sooner, reducing tail.
        # If we pick an op that fits on a low-load machine, we balance load.
        
        # Let's use a composite score:
        # score = min_est_finish - 0.5 * remaining_work
        # This favors ops that finish early AND have lots of work left.
        
        score = min_est_finish - 0.5 * remaining_work
        
        if score < best_score:
            best_score = score
            best_idx = i
            best_machine = chosen_machine
    
    if best_idx == -1:
        return 0

    # Update the load of the chosen machine
    # We need to get the duration of the chosen machine for the chosen op
    chosen_op = ready_operations[best_idx]
    dur = chosen_op["machine_options"][best_machine]
    _update_machine_load(best_machine, dur)

    # Return tuple to specify machine
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
