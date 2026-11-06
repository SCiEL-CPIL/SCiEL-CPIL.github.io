"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 145.0  # seconds; safety margin before 150s kill
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = _time.monotonic()
    return _time.monotonic() - _start_time


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    Returns an index (int) into ready_operations.
    
    Strategy: Constraint-tightness first (fewest machine options = highest priority),
    tie-break by remaining work (more work = higher priority),
    then by remaining ops (fewer = higher priority to avoid blocking).
    """
    if len(ready_operations) == 0:
        return 0
    if len(ready_operations) == 1:
        return 0

    # Deadline guard
    if _elapsed() > _DEADLINE:
        return 0

    n = len(ready_operations)

    # Compute constraint tightness: fewer machine options = more constrained
    # Fewer options means higher priority (to prevent starvation)
    machine_counts = []
    remaining_works = []
    remaining_ops = []
    min_durations = []

    for op in ready_operations:
        mo = op.get("machine_options", {})
        machine_counts.append(len(mo))
        remaining_works.append(op.get("remaining_work", 0))
        remaining_ops.append(op.get("remaining_ops", 0))
        # Minimum duration across available machines
        if mo:
            min_durations.append(min(mo.values()))
        else:
            min_durations.append(0)

    # Score: primary = -machine_count (fewer options = higher priority)
    # secondary = -remaining_work (more work = higher priority)  
    # tertiary = -remaining_ops (fewer ops left = higher priority, less blocking)
    # quaternary = min_duration (shorter = higher priority, less idle time)
    
    # We want to maximize priority, so:
    # - machine_count: lower is better (more constrained) -> use -count
    # - remaining_work: higher is better -> use +work
    # - remaining_ops: lower is better (less blocking) -> use -ops
    # - min_duration: lower is better (less delay) -> use -duration

    # Combine into a single score vector for argmax
    # Normalize to comparable scales
    max_work = max(remaining_works) if max(remaining_works) > 0 else 1
    max_ops = max(remaining_ops) if max(remaining_ops) > 0 else 1
    max_dur = max(min_durations) if max(min_durations) > 0 else 1

    scores = []
    for i in range(n):
        # Constraint tightness dominates
        tightness = -machine_counts[i]
        # Work load
        work_norm = remaining_works[i] / max_work
        # Remaining ops (fewer = less blocking)
        ops_norm = -remaining_ops[i] / max_ops
        # Duration (shorter = better)
        dur_norm = -min_durations[i] / max_dur

        score = tightness * 10 + work_norm * 5 + ops_norm * 3 + dur_norm * 1
        scores.append(score)

    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
