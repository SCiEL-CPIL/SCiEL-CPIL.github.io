"""Job-Shop Scheduling dispatch rule."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Track elapsed time for deadline safety
_start_time = _time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill

def dispatch(ready_operations, time):
    """Pick which ready operation to start next.
    
    Uses a composite score combining:
    - Shortest processing time (SPT) on best machine
    - Machine congestion (fewer pending ops is better)
    - Job urgency (less remaining work = more urgent)
    - Weighted combination tuned for makespan minimization
    """
    # Deadline guard
    if _time.time() - _start_time > _DEADLINE:
        # Fallback: just pick first
        return 0
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    scores = []
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        # Find the machine with shortest processing time
        min_duration = min(machine_options.values())
        min_machine = min(machine_options, key=machine_options.get)
        
        # SPT component: prefer shorter processing times
        spt_score = min_duration
        
        # Machine congestion: count how many other ready ops want this machine
        machine_load = 0
        for j, other_op in enumerate(ready_operations):
            if j != i and min_machine in other_op["machine_options"]:
                machine_load += 1
        
        # Job urgency: less remaining work = more urgent (higher score = better)
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Composite score (lower is better)
        # Weight SPT heavily, penalize machine congestion, add urgency
        # Normalize-ish: SPT dominates, congestion secondary
        score = spt_score * 2.0 + machine_load * 0.5 - remaining_work * 0.1
        
        scores.append(score)
    
    # Pick the operation with the lowest composite score (best)
    best_idx = int(np.argmin(scores))
    return best_idx
# EVOLVE-BLOCK-END
