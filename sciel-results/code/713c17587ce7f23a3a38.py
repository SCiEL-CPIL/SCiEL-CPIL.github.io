"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_START = _time.time()
_DEADLINE = 140.0  # Safe margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule: Normalized Shortest Processing Time (NSPT).
    
    For each ready operation, compute the minimum duration across its machine
    options, then normalize by remaining_work. Pick the operation with the
    lowest ratio (min_duration / (remaining_work + 1)).
    
    Rationale:
    - Short operations free up machines quickly (SPT benefit).
    - Normalizing by remaining_work ensures we don't starve jobs with lots
      of work left (MWRT benefit).
    - The +1 avoids division by zero and slightly penalizes jobs with no
      remaining work (which shouldn't happen for ready ops, but safe).
    """
    # Deadline guard: if we're close to the limit, fall back to a simple rule
    if _time.time() - _START > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    best_idx = 0
    best_score = float('inf')

    for i, op in enumerate(ready_operations):
        # Minimum duration across machine options
        durations = op["machine_options"].values()
        min_dur = min(durations) if durations else 1.0

        remaining_work = op.get("remaining_work", 0)
        if remaining_work is None or remaining_work < 0:
            remaining_work = 0

        # Composite score: lower is better
        score = min_dur / (remaining_work + 1.0)

        if score < best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
