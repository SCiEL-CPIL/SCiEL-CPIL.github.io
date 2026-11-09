"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Module-level deadline guard: we must finish within 120s per evaluation.
# The dispatch function is called many times per instance; we track total
# elapsed time across calls to ensure we don't exceed the budget.
_START_TIME = _time.monotonic()
TIME_BUDGET = 115.0  # safety margin before 120s limit


def _elapsed():
    return _time.monotonic() - _START_TIME


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Strategy: 
    Model the job-shop as a bipartite graph. For each ready operation, 
    select the machine that minimizes the estimated increase in critical path.
    Approximation: Minimize duration on the chosen machine to unblock the job sooner.
    Score operations by Urgency (Remaining Work) / Efficiency (1/Duration).
    Returns (index, machine) to explicitly assign the best machine.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    
    # Check time budget - if we're running low, fall back to fast MWL
    if _elapsed() > TIME_BUDGET - 5.0:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    rw = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    
    # Pre-compute best machine and duration for each op
    best_machines = [None] * n
    best_durs = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            # Find machine with minimum duration
            best_m, best_d = min(mo.items(), key=lambda item: item[1])
            best_machines[i] = best_m
            best_durs[i] = best_d
        else:
            best_machines[i] = 0
            best_durs[i] = 0.0

    # Score: Remaining Work / (Best Duration + 1)
    # High RW (urgency) and Low Duration (efficiency) yield high score.
    # The +1.0 prevents division by zero and smooths the function.
    raw_scores = rw / (best_durs + 1.0)
    
    max_score = raw_scores.max()
    if max_score > 0:
        scores = raw_scores / max_score
    else:
        scores = np.zeros(n)
        
    best_idx = int(np.argmax(scores))
    best_machine = best_machines[best_idx]
    
    # Return tuple (index, machine) as allowed by interface
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
