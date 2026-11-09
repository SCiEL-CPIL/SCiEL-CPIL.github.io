"""Improved job-shop dispatch rule: simplified critical path with flexibility."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START_TIME = _time.monotonic()
_DEADLINE = 140.0


def _compute_score(ready_operations, time):
    """Compute a composite score for each ready operation. Higher = better to pick.
    
    Strategy: 
    1. Estimate the critical path contribution: min_duration + remaining_work_after
    2. Add flexibility bonus: more machine options = less contention
    3. SPT: shorter operations reduce blocking
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Extract features
    rw = np.zeros(n, dtype=np.float64)      # remaining work
    min_dur = np.zeros(n, dtype=np.float64) # min duration across machine options
    n_opts = np.zeros(n, dtype=np.float64)  # number of machine options

    for i, op in enumerate(ready_operations):
        rw[i] = op.get("remaining_work", 0)
        
        machine_options = op.get("machine_options", {})
        if machine_options:
            durations = np.array(list(machine_options.values()), dtype=np.float64)
            min_dur[i] = float(np.min(durations))
            n_opts[i] = float(len(machine_options))
        else:
            min_dur[i] = 1.0
            n_opts[i] = 1.0

    # Critical path estimate: duration + remaining work after this op
    # remaining_work_after = max(rw - min_dur, 0)
    # cp_contrib = min_dur + max(rw - min_dur, 0) = max(rw, min_dur)
    # But this simplifies to: if rw > min_dur, cp = rw; else cp = min_dur
    # Actually: cp_contrib = min_dur + max(rw - min_dur, 0)
    # If rw >= min_dur: cp_contrib = min_dur + rw - min_dur = rw
    # If rw < min_dur: cp_contrib = min_dur + 0 = min_dur
    # So cp_contrib = max(rw, min_dur)
    
    # But we want to minimize the INCREASE in makespan. The operation adds min_dur
    # to the machine's load, and unblocks the job's remaining work.
    # The critical path through this operation is: min_dur + (remaining work of job after this op)
    # remaining work after = max(rw - min_dur, 0)
    # So cp = min_dur + max(rw - min_dur, 0) = max(rw, min_dur)
    
    # We want to MINIMIZE cp, so we invert
    cp = np.maximum(rw, min_dur)
    
    # Normalize to [0, 1]
    def _normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        rng = mx - mn
        if rng < 1e-9:
            return np.full_like(arr, 0.5)
        return (arr - mn) / rng
    
    # Inverse critical path: lower cp is better
    norm_inv_cp = _normalize(-cp)
    
    # SPT: shorter min_duration is better
    norm_spt = _normalize(-min_dur)
    
    # Flexibility: more options is better
    norm_flex = _normalize(n_opts)
    
    # Remaining work urgency: jobs with more work remaining are more urgent
    norm_rw = _normalize(rw)
    
    # Composite score
    # Primary: minimize critical path contribution
    # Secondary: SPT (shorter ops first)
    # Tertiary: flexibility (more options = less contention)
    # Minor: remaining work urgency
    
    score = (
        1.0 * norm_inv_cp      # PRIMARY: minimize estimated critical path
        + 0.5 * norm_spt       # SPT: shorter operations
        + 0.3 * norm_flex      # Flexibility: more machine options
        + 0.2 * norm_rw        # Urgency: more remaining work
    )
    
    return int(np.argmax(score))


def dispatch(ready_operations, time):
    """Dispatch rule for job-shop scheduling.

    Returns the index of the ready operation to start next.
    """
    if not ready_operations:
        return 0

    # Deadline guard
    if _time.monotonic() - _START_TIME > _DEADLINE - 5.0:
        # Fallback: pick the one with most remaining work
        return int(np.argmax([op.get("remaining_work", 0) for op in ready_operations]))

    return _compute_score(ready_operations, time)
# EVOLVE-BLOCK-END
