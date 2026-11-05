"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def _compute_op_score(op, time):
    """
    Compute a priority score for a ready operation.
    Higher score means higher priority (start first).
    
    Heuristic components:
    1. Remaining work (primary): jobs with more total remaining work are more critical.
    2. Minimum duration: operations that can finish quickly unlock subsequent operations.
    3. Number of remaining ops: jobs with fewer remaining ops are closer to completion.
    4. Machine load consideration: prefer machines that are less loaded (if we could tell).
    """
    work = op["remaining_work"]
    machine_options = op["machine_options"]
    
    if not machine_options:
        return 0.0
    
    durations = list(machine_options.values())
    min_dur = min(durations)
    max_dur = max(durations)
    avg_dur = sum(durations) / len(durations)
    
    remaining_ops = op.get("remaining_ops", 0)
    
    # Weighted composite score
    # Primary: remaining work (normalized by scale)
    # Secondary: shorter min duration is better (finish faster, unlock next op)
    # Tertiary: fewer remaining ops means closer to done (urgency)
    
    score = 0.0
    score += 1.0 * work  # Most work remaining
    score += 0.3 / (min_dur + 1e-9)  # Prefer shorter operations
    score += 0.2 / (remaining_ops + 1e-9)  # Prefer jobs closer to completion
    
    # Penalty for very long max duration relative to min (uncertainty)
    if max_dur > min_dur:
        score -= 0.1 * (max_dur - min_dur) / (min_dur + 1e-9)
    
    return score


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Returns: index of the ready operation to start.
    """
    if not ready_operations:
        return 0
    
    if len(ready_operations) == 1:
        return 0
    
    # Compute scores for all ready operations
    scores = [_compute_op_score(op, time) for op in ready_operations]
    
    # Return index of max score
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
