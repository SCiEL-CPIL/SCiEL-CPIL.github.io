"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_DEADLINE = 140.0


def _simple_dispatch(ready_operations, time):
    """Fallback simple dispatch: most remaining work first."""
    if not ready_operations:
        return 0
    return int(np.argmax([op["remaining_work"] for op in ready_operations]))


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy: Prioritize operations that help minimize makespan.
    In open-shop, makespan >= max(max_job_remaining, max_machine_load).
    
    Key heuristics:
    1. Jobs with more remaining work should be prioritized (they're on the critical path)
    2. Operations with shorter processing times should be prioritized (frees machines faster)
    3. Assign to the machine that minimizes the resulting machine load
    
    Score: remaining_work * weight1 + (1/min_duration) * weight2
    This favors operations that have lots of remaining work but can be done quickly.
    """
    if not ready_operations:
        return 0
    
    if _time.time() > _DEADLINE:
        return _simple_dispatch(ready_operations, time)
    
    n = len(ready_operations)
    
    scores = []
    for op in ready_operations:
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            scores.append(0)
            continue
        
        min_dur = min(machine_options.values())
        avg_dur = sum(machine_options.values()) / len(machine_options)
        
        if remaining_work <= 0 and remaining_ops > 0:
            remaining_work = remaining_ops * avg_dur
        
        if remaining_work <= 0:
            scores.append(0)
            continue
        
        # Primary: remaining work (jobs with more work are more critical)
        # Secondary: 1/min_duration (shorter operations free up machines faster)
        # The ratio remaining_work / min_dur favors operations that have lots of work
        # but can be completed quickly on some machine
        
        score = remaining_work * 1.0 + (1.0 / max(min_dur, 0.01)) * 1.0
        
        scores.append(score)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
