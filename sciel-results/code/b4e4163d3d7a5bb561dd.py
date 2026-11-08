"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
import random as _random

# Global deadline guard to ensure we finish well within the 120s evaluation limit
_START_TIME = _time.time()
_HARD_LIMIT = 110.0  # seconds
_SOFT_MARGIN = 5.0   # seconds buffer before hard limit


def _check_deadline():
    """Return True if we are approaching the hard limit."""
    return (_time.time() - _START_TIME) > (_HARD_LIMIT - _SOFT_MARGIN)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    Uses a hybrid heuristic combining multiple scheduling rules to minimize makespan.
    """
    if not ready_operations:
        return 0
    
    # Deadline check - if we're close to the limit, use a simple fast heuristic
    if _check_deadline():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n = len(ready_operations)
    
    # Compute various metrics for each ready operation
    scores = []
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        # Minimum duration among available machines
        min_duration = min(machine_options.values()) if machine_options else 0
        max_duration = max(machine_options.values()) if machine_options else 0
        
        # Average duration
        avg_duration = np.mean(list(machine_options.values())) if machine_options else 0
        
        # Number of machine options (flexibility)
        num_options = len(machine_options)
        
        # Work per remaining op (urgency)
        work_per_op = remaining_work / max(remaining_ops, 1)
        
        # Combine multiple factors into a score
        # Higher score = more urgent / better to schedule now
        
        # Factor 1: Total remaining work (longest processing time first)
        f1 = remaining_work
        
        # Factor 2: Minimum possible completion time (shortest processing time)
        f2 = min_duration
        
        # Factor 3: Work density (work per remaining operation)
        f3 = work_per_op
        
        # Factor 4: Flexibility (fewer options = more constrained, schedule sooner)
        f4 = 1.0 / max(num_options, 1)
        
        # Factor 5: Ratio of min to max duration (tighter range = less variability)
        f5 = min_duration / max(max_duration, 1e-9)
        
        # Weighted combination - tuned for open-shop scheduling
        # Prioritize operations with more remaining work, shorter min duration,
        # and higher work density
        score = (
            1.0 * f1 +          # Remaining work
            0.5 * f2 +           # Min duration (shorter is better to start early)
            0.3 * f3 +           # Work per op
            0.2 * f4 +           # Constrained machines first
            -0.1 * f5            # Slight penalty for very tight ranges
        )
        
        scores.append(score)
    
    scores = np.array(scores)
    
    # Add small random tie-breaker to avoid deterministic bias
    if n > 1:
        tie_breaker = _random.random() * 0.01
        scores = scores + tie_breaker * np.ones(n)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
