"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE_MARGIN = 10.0  # seconds of safety margin before the 120s wall-clock kill
_START_TIME = _time.time()


def _elapsed():
    return _time.time() - _START_TIME


def _deadline_exceeded():
    return _elapsed() > (120.0 - _DEADLINE_MARGIN)


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Strategy: Bottleneck mitigation via constraint tightness.
    1. Prioritize operations with fewest remaining machine options (highest constraint).
    2. Break ties by longest remaining work (to keep machines busy).
    3. Break further ties by shortest minimum processing time (to reduce idle gaps).
    
    Returns the index of the ready operation to start next.
    Optionally could return (index, machine) but we stick to index for compatibility.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # If we're close to the deadline, use the fastest heuristic
    if _deadline_exceeded():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Compute scores for each ready operation
    # We want to MINIMIZE the score (pick the smallest), or MAXIMIZE a priority.
    # Let's define a priority where higher is better.
    
    # Constraint tightness: fewer machine options = higher priority
    # Machine options count
    machine_counts = [len(op["machine_options"]) for op in ready_operations]
    
    # Remaining work: more work = higher priority (to avoid starvation of large jobs)
    remaining_work = [op["remaining_work"] for op in ready_operations]
    
    # Minimum processing time among options: shorter = potentially less blocking?
    # Actually, for makespan, we often want to start operations that have short min times
    # to keep flow tight, OR long times to get them out of the way.
    # Let's look at the "shortest possible completion" for this op.
    min_durations = [min(op["machine_options"].values()) for op in ready_operations]
    
    # Heuristic:
    # Primary: Fewest machine options (constraint) -> lower is better, so invert for max
    # Secondary: Most remaining work -> higher is better
    # Tertiary: Shortest min duration -> lower is better, so invert for max? 
    # Actually, if we pick an op with a very short min duration, it finishes quickly, freeing up resources.
    # But if we pick an op with a very long min duration, it occupies a machine for a long time.
    # In JSS, SPT (Shortest Processing Time) is often good for flow time, LPT for makespan?
    # For makespan, we generally want to keep the critical path moving.
    
    # Let's try a weighted score.
    # Normalize the values to [0, 1] for combination?
    # Or just use a lexicographic tuple for sorting?
    # Lexicographic is safer to avoid scaling issues.
    
    # Sort key: (num_machine_options ASC, remaining_work DESC, min_duration ASC)
    # We want to pick the "best" one.
    # Let's create a list of tuples and find the min/max.
    
    # Key for sorting (we want to minimize this tuple):
    # 1. num_machine_options (lower is better)
    # 2. -remaining_work (lower is better, i.e., higher work is better)
    # 3. min_duration (lower is better)
    
    keys = []
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        num_opts = len(opts)
        work = op["remaining_work"]
        min_dur = min(opts.values())
        # We want to minimize: (num_opts, -work, min_dur)
        keys.append((num_opts, -work, min_dur))
    
    # Find the index with the minimum key
    best_idx = int(np.argmin(keys, axis=0))
    
    return best_idx
# EVOLVE-BLOCK-END
