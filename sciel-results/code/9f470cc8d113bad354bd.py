"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: the evaluator kills after 150s wall-clock.
# We keep a safety margin.
_DEADLINE_SECONDS = 145.0
_start_wall = _time.monotonic()


def _elapsed():
    return _time.monotonic() - _start_wall


def _is_deadline_exceeded():
    return _elapsed() >= _DEADLINE_SECONDS


def _compute_score(ready_operations, time):
    """
    Score each ready operation using a combination of:
    - Minimum processing time (SPT)
    - Remaining work (MW)
    - Number of remaining ops (fewer is better for finishing soon)
    
    We use a weighted combination. The key insight for job-shop:
    - SPT (shortest processing time first) is a strong heuristic
    - But we also want to avoid blocking machines for too long
    - Remaining work gives context about how much is left
    
    We'll use: score = min_duration * 1.0 + remaining_work * 0.1 - remaining_ops * 0.05
    Actually, let's think more carefully.
    
    For job-shop scheduling, the classic dispatching rules are:
    - SPT: Shortest Processing Time
    - LPT: Longest Processing Time  
    - MW: Most Work Remaining
    - MWCR: Most Work Remaining with Critical Ratio
    
    A good combined heuristic: prioritize operations that have short processing times
    on their assigned machine, but also consider the overall remaining work.
    
    Let's use a score that favors:
    1. Short processing time (primary)
    2. Less remaining work (secondary) - to finish jobs sooner
    3. Fewer remaining ops (tertiary)
    
    score = min_proc_time + 0.2 * remaining_work / max(1, remaining_ops)
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    
    scores = np.empty(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        # Find the minimum duration across available machines
        min_dur = min(machine_options.values())
        remaining_work = op["remaining_work"]
        remaining_ops = max(1, op["remaining_ops"])
        
        # Combined score: lower is better
        # Primary: shortest processing time
        # Secondary: normalize remaining work by remaining ops
        scores[i] = min_dur + 0.15 * (remaining_work / remaining_ops)
    
    # Return index of minimum score
    return int(np.argmin(scores))


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Args:
        ready_operations: list of dicts, each with:
            - job: job id
            - op: operation id
            - machine_options: {machine: duration}
            - remaining_ops: number of ops remaining in the job
            - remaining_work: total remaining processing work
        time: current simulation time
    
    Returns:
        index of the ready operation to start
    """
    if _is_deadline_exceeded():
        # Safety fallback: just pick the first one
        return 0
    
    if not ready_operations:
        return 0
    
    if len(ready_operations) == 1:
        return 0
    
    # Use the combined scoring heuristic
    return _compute_score(ready_operations, time)
# EVOLVE-BLOCK-END
