"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Global deadline guard: 150s wall-clock limit, safety margin 5s
_DEADLINE = _time.time() + 145.0


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Strategy: A composite heuristic that considers:
    1. Minimum remaining processing time on the selected machine (shortest job first)
    2. Remaining work of the job (prioritize jobs with more work left to avoid starvation)
    3. Number of remaining operations (jobs closer to completion get priority)
    
    For each ready operation, we evaluate all machine options and pick the one
    with minimum duration. Then we score the operation based on a weighted combination.
    """
    # Deadline check
    if _time.time() > _DEADLINE:
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    scores = []
    for i, op in enumerate(ready_operations):
        # Find the minimum duration across all machine options
        machine_options = op["machine_options"]
        if machine_options:
            min_duration = min(machine_options.values())
        else:
            min_duration = 0

        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]

        # Composite score:
        # - Lower min_duration is better (shortest processing time first)
        # - Higher remaining_work is better (prioritize jobs with more work)
        # - Lower remaining_ops is better (prioritize jobs closer to completion)
        
        # Normalize-ish: use weighted sum
        # SPT component: lower is better -> negate
        # Work component: higher is better
        # Ops component: lower is better -> negate
        
        score = (
            -1.0 * min_duration +          # SPT: prefer shorter processing
            0.5 * remaining_work -          # Prioritize more remaining work
            0.3 * remaining_ops             # Prefer fewer remaining ops
        )
        scores.append(score)

    # Pick the operation with the highest composite score
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
