"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_DEADLINE_SECONDS = 145.0  # Safety margin before 150s kill


def _compute_op_score(op, time):
    """
    Compute a priority score for a ready operation.
    Higher score means higher priority (start first).
    
    Strategy: Prioritize operations from jobs with the most remaining work,
    favoring shorter operations on less-contended machines to reduce makespan.
    """
    work = op.get("remaining_work", 0)
    machine_options = op.get("machine_options", {})
    
    if not machine_options:
        return 0.0
    
    durations = list(machine_options.values())
    min_dur = min(durations)
    max_dur = max(durations)
    remaining_ops = op.get("remaining_ops", 0)
    
    # Primary: total remaining work (critical for makespan)
    score = 1.0 * work
    
    # Secondary: prefer shorter minimum duration (faster completion)
    score += 0.5 / (min_dur + 1e-9)
    
    # Tertiary: prefer jobs closer to completion (fewer remaining ops)
    score += 0.3 / (remaining_ops + 1e-9)
    
    # Penalty for high variance in machine options (uncertainty)
    if max_dur > min_dur:
        score -= 0.05 * (max_dur - min_dur) / (min_dur + 1e-9)
    
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
    
    # Deadline guard
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        return 0
    
    # Compute scores for all ready operations
    scores = [_compute_op_score(op, time) for op in ready_operations]
    
    # Return index of max score
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
