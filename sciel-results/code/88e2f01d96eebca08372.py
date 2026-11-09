"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from scipy.optimize import linear_sum_assignment

# Track elapsed time to enforce deadline
_start_time = None
# Track machine loads to inform dispatch decisions
_machine_loads = {}
# Track job remaining work for urgency
_job_remaining_work = {}

def _elapsed_ok(deadline_margin=5.0):
    """Check if we're still within the time budget."""
    global _start_time
    if _start_time is None:
        _start_time = _time.time()
    return (_time.time() - _start_time) < (150.0 - deadline_margin)


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy: Use a load-aware heuristic that prioritizes:
    1. Operations on machines with lower current load (to balance load)
    2. Operations with high remaining work (to finish jobs quickly)
    3. Operations with fewer machine options (to reduce flexibility loss)
    
    We use a weighted score that combines these factors, with machine load
    tracking to avoid bottlenecks.
    """
    global _machine_loads, _job_remaining_work
    
    if not ready_operations:
        return 0
    
    # Check deadline - if close to timeout, use fast heuristic
    if not _elapsed_ok(deadline_margin=10.0):
        # Fast fallback: most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n_ops = len(ready_operations)
    scores = np.zeros(n_ops)
    
    # Update job remaining work tracking
    for op in ready_operations:
        job = op["job"]
        if job not in _job_remaining_work:
            _job_remaining_work[job] = op["remaining_work"]
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        # Factor 1: Remaining work (higher is better)
        work_score = remaining_work
        
        # Factor 2: Number of remaining operations (jobs with fewer remaining ops should be prioritized)
        ops_score = 1.0 / max(remaining_ops, 1)
        
        # Factor 3: Flexibility - operations with fewer machine options should be prioritized
        # (to avoid getting stuck later)
        num_options = len(machine_options)
        flex_score = 1.0 / max(num_options, 1)
        
        # Factor 4: Machine load balancing - prefer operations that can go to less-loaded machines
        # Estimate minimum load among available machines
        if machine_options:
            min_machine_load = min(_machine_loads.get(m, 0) for m in machine_options.keys())
            max_machine_load = max(_machine_loads.get(m, 0) for m in machine_options.keys())
            # Lower load is better, so invert
            load_score = 1.0 / (1.0 + min_machine_load)
        else:
            load_score = 0.0
        
        # Factor 5: Minimum duration among options (shorter processing times first)
        min_duration = min(machine_options.values()) if machine_options else 0
        duration_score = 1.0 / max(min_duration, 0.1)
        
        # Combine with weights
        # Weight: work=0.25, ops=0.15, flexibility=0.1, load=0.3, duration=0.2
        scores[i] = (0.25 * work_score + 
                     0.15 * ops_score + 
                     0.1 * flex_score + 
                     0.3 * load_score + 
                     0.2 * duration_score)
    
    # If scores are all zero or invalid, fallback
    if not np.any(scores > 0):
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Select the operation with the highest score
    best_idx = int(np.argmax(scores))
    
    # Update machine load tracking for the selected operation
    # We don't know which machine will be chosen, so we estimate
    # by adding the minimum duration to the least-loaded machine
    best_op = ready_operations[best_idx]
    if best_op["machine_options"]:
        # Find the least-loaded machine among options
        min_load_machine = min(best_op["machine_options"].keys(), 
                              key=lambda m: _machine_loads.get(m, 0))
        min_duration = min(best_op["machine_options"].values())
        _machine_loads[min_load_machine] = _machine_loads.get(min_load_machine, 0) + min_duration
    
    return best_idx
# EVOLVE-BLOCK-END
