"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from collections import defaultdict

# Deadline guard: the evaluator kills after 150s wall-clock.
_T0 = _time.perf_counter()
_DEADLINE = 140.0  # safety margin of 10s

# Track estimated machine loads across dispatch calls
_machine_loads = defaultdict(float)
_total_time_seen = 0.0


def _elapsed_ok():
    return (_time.perf_counter() - _T0) < _DEADLINE


def dispatch(ready_operations, time):
    """
    Dispatch rule using load-balancing assignment approach.
    
    For each ready operation, we estimate the makespan impact if we start it
    on its best available machine (considering current machine load).
    We pick the operation that minimizes the estimated maximum completion time
    across all machines, with tie-breaking by remaining work urgency.
    """
    global _machine_loads, _total_time_seen
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Quick deadline check - fall back to simple heuristic
    if not _elapsed_ok():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Update total time seen (for normalization)
    if time > _total_time_seen:
        _total_time_seen = time
    
    # For each operation, find the machine that minimizes estimated makespan
    # when this operation is assigned to it
    best_score = float('inf')
    best_idx = 0
    
    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        ro = op["remaining_ops"]
        mo = op["machine_options"]
        
        if not mo:
            # No machine options - use remaining work as priority
            score = -rw  # lower is better in our comparison
            if score < best_score:
                best_score = score
                best_idx = i
            continue
        
        # For this operation, find the best machine assignment
        # that minimizes the resulting makespan contribution
        min_makespan_contrib = float('inf')
        
        for machine, duration in mo.items():
            current_load = _machine_loads.get(machine, 0.0)
            
            # If we assign this op to this machine:
            # - The machine's new completion time would be current_load + duration
            # - But we also need to consider that other ops might follow on this machine
            # 
            # Key insight: we want to minimize the makespan, which is dominated
            # by the most loaded machine. So we prefer assignments that:
            # 1. Don't create a new maximum load
            # 2. Balance loads across machines
            # 3. Prioritize jobs with more remaining work (urgency)
            
            new_machine_load = current_load + duration
            
            # Penalty for creating high load on a machine
            # Normalize by total time to make scale-invariant
            norm_time = max(_total_time_seen, 1.0)
            load_penalty = (new_machine_load / norm_time) ** 2
            
            # Urgency bonus: jobs with more remaining work should go first
            urgency = rw / max(ro, 1.0)
            
            # Combined score: we want to minimize load_penalty, maximize urgency
            # Using a weighted combination
            score = load_penalty * 1.0 - urgency * 0.3
            
            if score < min_makespan_contrib:
                min_makespan_contrib = score
        
        if min_makespan_contrib < best_score:
            best_score = min_makespan_contrib
            best_idx = i
    
    return best_idx
# EVOLVE-BLOCK-END
