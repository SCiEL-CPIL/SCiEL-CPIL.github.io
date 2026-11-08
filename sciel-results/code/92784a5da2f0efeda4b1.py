"""Rewritten dispatch rule for job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time

# Global deadline guard
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s limit


def _is_deadline_exceeded():
    return _time.time() > _DEADLINE


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Uses Minimum Remaining Processing Time (MRPT) as primary criterion,
    with tiebreaking by fewer remaining operations to complete jobs sooner.
    
    Returns the index of the chosen operation.
    """
    if not ready_operations:
        return 0
    
    if _is_deadline_exceeded():
        # Fallback: pick the operation with minimum processing time
        min_idx = 0
        min_dur = float('inf')
        for i, op in enumerate(ready_operations):
            min_op_dur = min(op["machine_options"].values())
            if min_op_dur < min_dur:
                min_dur = min_op_dur
                min_idx = i
        return int(min_idx)
    
    n = len(ready_operations)
    
    # Find the operation with minimum remaining processing time
    best_idx = 0
    best_min_dur = float('inf')
    best_rem_ops = float('inf')
    
    for i, op in enumerate(ready_operations):
        min_dur = min(op["machine_options"].values())
        rem_ops = op["remaining_ops"]
        
        # Primary: minimize processing time
        # Secondary: minimize remaining operations (tiebreaker)
        if min_dur < best_min_dur:
            best_min_dur = min_dur
            best_rem_ops = rem_ops
            best_idx = i
        elif min_dur == best_min_dur:
            if rem_ops < best_rem_ops:
                best_rem_ops = rem_ops
                best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
