import numpy as np
import time as _time_module

_START_TIME = _time_module.perf_counter()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling (makespan minimization).
    
    Uses a refined multi-criteria scoring approach:
    1. Urgency: Prioritizes jobs with more remaining work (remaining_work if available, else remaining_ops).
    2. Speed: Penalizes longer processing times using a power law (gamma=1.5) to strongly favor 
       operations that can be completed quickly on their best available machine.
    3. Tie-breaking: Prefers shorter processing times (SPT) to reduce machine blocking.
    
    This aims to balance the "Most Work Remaining" (to prevent starvation of large jobs) 
    with "Shortest Processing Time" (to keep machines busy and reduce idle time).
    """
    if not ready_operations:
        return 0

    if _time_module.perf_counter() > _DEADLINE:
        return 0

    best_index = 0
    best_score = -float('inf')
    best_tie_breaker = float('inf')

    n_ops = len(ready_operations)
    gamma = 1.5  # Exponent for duration penalty. >1 favors short jobs more strongly.

    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            min_duration = 1.0
        else:
            durations = list(machine_options.values())
            if not durations:
                min_duration = 1.0
            else:
                min_duration = min(durations)
                if min_duration <= 0:
                    min_duration = 1e-9

        # Determine urgency metric
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 1)
        
        if remaining_work is not None and remaining_work > 0:
            urgency = remaining_work
        else:
            # If remaining_work is not a good proxy, fall back to op count
            # We assume 1 unit of work per op for scaling if needed, 
            # but since we are comparing relative scores within a batch, 
            # raw op count is a valid relative urgency measure.
            urgency = remaining_ops if remaining_ops > 0 else 1

        if urgency <= 0:
            urgency = 1

        # Calculate primary score
        # Higher urgency and lower duration -> Higher score
        # Using power law on duration to amplify the benefit of short jobs
        score = urgency / (min_duration ** gamma)

        # Tie-breaking: Prefer shorter duration (SPT) if scores are close
        # This helps in clearing the queue of fast jobs
        tie_breaker = min_duration

        epsilon = 1e-9
        if score > best_score + epsilon:
            best_score = score
            best_tie_breaker = tie_breaker
            best_index = i
        elif abs(score - best_score) <= epsilon:
            if tie_breaker < best_tie_breaker:
                best_tie_breaker = tie_breaker
                best_index = i

    return int(best_index)
