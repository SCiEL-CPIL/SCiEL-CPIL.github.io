"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np


# Module-level deadline guard
_START_TIME = None
_TIME_LIMIT = 135.0  # 15s safety margin before 150s kill
_DEADLINE_REACHED = False


def _check_deadline():
    global _START_TIME, _DEADLINE_REACHED
    if _START_TIME is None:
        _START_TIME = _time.monotonic()
    if _time.monotonic() - _START_TIME > _TIME_LIMIT:
        _DEADLINE_REACHED = True
    return _DEADLINE_REACHED


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy: Return (index, machine) to explicitly assign each operation to its fastest machine.
    This allows the simulator to place operations on the machine with the shortest processing time,
    directly reducing makespan.
    
    Score combines:
    1. Remaining work (total) - prefer jobs with more remaining work
    2. Minimum machine option duration via soft-min (log-sum-exp) - smoother surrogate
    3. Number of machine options (flexibility) - prefer operations with more choices
    4. Remaining ops - prefer jobs with fewer remaining ops (to complete jobs)
    
    The soft-min via log-sum-exp provides a smooth approximation to the minimum duration,
    making the scoring objective smoother and more differentiable while capturing the
    benefit of operations that can be placed on fast machines.
    """
    if not ready_operations:
        return 0
    
    if _check_deadline():
        # Fallback: return first operation with its fastest machine
        op = ready_operations[0]
        mo = op["machine_options"]
        if isinstance(mo, dict) and mo:
            best_machine = min(mo, key=mo.get)
            return (0, best_machine)
        return 0
    
    n = len(ready_operations)
    
    # Compute scores using numpy for efficiency
    scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Get machine options
        machine_options = op["machine_options"]
        if isinstance(machine_options, dict) and machine_options:
            durations = np.array(list(machine_options.values()), dtype=np.float64)
            n_machines = len(machine_options)
            
            # Soft-min via log-sum-exp: approximates min(durations) smoothly
            # Using a small temperature for a sharp approximation
            temp = 0.5
            log_sum_exp = np.log(np.sum(np.exp(-durations / temp))) - np.log(n_machines)
            soft_min = -temp * log_sum_exp
            
            # Flexibility bonus: more machine options means more scheduling flexibility
            flexibility = np.log1p(n_machines)  # log(1 + n_machines)
            
            # Find the best (fastest) machine for explicit assignment
            best_machine = min(machine_options, key=machine_options.get)
        else:
            soft_min = 0.0
            flexibility = 0.0
            best_machine = None
        
        # Weighted score: 
        # - Emphasize remaining work (keep busy jobs progressing)
        # - Penalize long minimum duration (shorter ops are easier to slot in)
        # - Reward flexibility (more machine options = more scheduling options)
        # - Slightly favor jobs with fewer remaining ops (complete jobs sooner)
        scores[i] = (
            remaining_work * 1.0
            - soft_min * 0.4
            + flexibility * 0.3
            - remaining_ops * 0.2
        )
    
    best_idx = int(np.argmax(scores))
    
    # Return (index, machine) to explicitly assign to the fastest machine
    op = ready_operations[best_idx]
    mo = op["machine_options"]
    if isinstance(mo, dict) and mo:
        best_machine = min(mo, key=mo.get)
        return (best_idx, best_machine)
    
    return best_idx
# EVOLVE-BLOCK-END
