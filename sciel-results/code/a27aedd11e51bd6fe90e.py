import numpy as np
import time as _time_module

# EVOLVE-BLOCK-START
_START_TIME = _time_module.perf_counter()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for Job-Shop Scheduling.
    
    Combines the structural WSPT-like heuristic (remaining_ops / min_duration)
    from inspiration 1 with a refined composite scoring mechanism that
    incorporates remaining_work and average duration as secondary signals.
    
    Primary score: remaining_ops / min_duration (maximize)
    Refinement: incorporate remaining_work to handle critical-path jobs and
    avg_duration for machine-load awareness.
    """
    if not ready_operations:
        return 0

    # Deadline guard: if we're getting close to the limit, use fast fallback
    if _time_module.perf_counter() > _DEADLINE:
        # Fast fallback: just pick the one with most remaining work (seed behavior)
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)

    # Compute metrics for each ready operation
    min_durations = []
    avg_durations = []
    remaining_works = []
    remaining_ops_list = []

    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        if not machine_options:
            min_dur = 1.0
            avg_dur = 1.0
        else:
            durations = list(machine_options.values())
            min_dur = min(durations)
            avg_dur = sum(durations) / len(durations)
            if min_dur <= 0:
                min_dur = 1e-9
            if avg_dur <= 0:
                avg_dur = 1e-9

        min_durations.append(min_dur)
        avg_durations.append(avg_dur)

        rem_work = op.get("remaining_work", 0)
        if rem_work <= 0:
            rem_work = min_dur
        remaining_works.append(rem_work)

        rem_ops = op.get("remaining_ops", 1)
        if rem_ops <= 0:
            rem_ops = 1
        remaining_ops_list.append(rem_ops)

    # Convert to numpy arrays for vectorized computation
    min_durations = np.array(min_durations, dtype=np.float64)
    avg_durations = np.array(avg_durations, dtype=np.float64)
    remaining_works = np.array(remaining_works, dtype=np.float64)
    remaining_ops_arr = np.array(remaining_ops_list, dtype=np.float64)

    # Primary score: remaining_ops / min_duration (WSPT-like, maximize)
    # This was the core of inspiration 1's 1.0583 score
    primary_score = remaining_ops_arr / min_durations

    # Refinement: incorporate remaining_work as a secondary signal
    # Normalize remaining_work to [0,1] (lower is better for makespan)
    def normalize_lower_better(arr):
        """Normalize so that lower values get higher scores in [0,1]."""
        if len(arr) == 0:
            return arr
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        normed = 1.0 - (arr - mn) / (mx - mn)
        return normed

    # Normalize primary_score to [0,1]
    p_min, p_max = primary_score.min(), primary_score.max()
    if p_max - p_min < 1e-12:
        primary_norm = np.ones_like(primary_score) * 0.5
    else:
        primary_norm = (primary_score - p_min) / (p_max - p_min)

    # Normalize remaining_work (lower remaining work is better)
    work_norm = normalize_lower_better(remaining_works)

    # Weighted combination:
    # - Primary WSPT score gets most weight (it drove the 1.0583 improvement)
    # - Remaining work gets a smaller weight as a tie-breaker/refinement
    # The avg_duration term adds slight machine-load awareness
    avg_norm = normalize_lower_better(avg_durations)

    weights = np.array([0.65, 0.2, 0.15])
    total_score = weights[0] * primary_norm + weights[1] * work_norm + weights[2] * avg_norm

    # Return index of operation with highest priority score
    return int(np.argmax(total_score))
# EVOLVE-BLOCK-END
