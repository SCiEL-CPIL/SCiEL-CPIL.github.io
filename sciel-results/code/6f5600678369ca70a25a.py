"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: we must return well before the 150s wall-clock limit.
_DEADLINE = 145.0  # seconds
_start_time = _time.time()


def _check_deadline():
    return (_time.time() - _start_time) > _DEADLINE


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Uses a multi-criteria weighted scoring approach that balances:
    1. Remaining work (urgency of job completion)
    2. Remaining operations count (job length)
    3. Minimum machine duration (to reduce immediate bottleneck)
    4. A look-ahead heuristic considering machine contention
    
    Returns index of the ready operation to start.
    """
    if _check_deadline():
        # Fallback: pick the operation with the most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Precompute features for each ready operation
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    
    # For each operation, find the minimum duration across its machine options
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else float('inf')
        for op in ready_operations
    ], dtype=float)
    
    # Normalize each feature to [0, 1] range for fair weighting
    # Use safe normalization to avoid division by zero
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)
    
    nw_norm = normalize(remaining_works)
    no_norm = normalize(remaining_ops)
    # For min duration, lower is better, so we invert
    md_norm = 1.0 - normalize(min_durations)
    
    # Weights: prioritize remaining work most, then remaining ops, then min duration
    # These weights are tuned to balance machine utilization vs job completion urgency
    w_rw = 0.45  # remaining work
    w_ro = 0.30  # remaining ops
    w_md = 0.25  # min machine duration (inverted)
    
    scores = w_rw * nw_norm + w_ro * no_norm + w_md * md_norm
    
    # Additional tie-breaking: prefer operations on machines with fewer options
    # (more constrained machines should be utilized to reduce future conflicts)
    num_machine_options = np.array([
        len(op["machine_options"]) if op["machine_options"] else 0
        for op in ready_operations
    ], dtype=float)
    
    # Fewer options = more constrained, give slight boost
    if num_machine_options.max() > num_machine_options.min():
        mo_norm = 1.0 - normalize(num_machine_options)
        scores += 0.05 * mo_norm
    
    # Handle ties deterministically: prefer lower job id, then lower op
    best_score = scores.max()
    candidates = [i for i in range(n) if abs(scores[i] - best_score) < 1e-9]
    
    if len(candidates) == 1:
        return int(candidates[0])
    
    # Tie-break: prefer job with more remaining work, then lower job number
    candidates_sorted = sorted(
        candidates,
        key=lambda i: (-remaining_works[i], ready_operations[i]["job"], ready_operations[i]["op"])
    )
    
    return int(candidates_sorted[0])
# EVOLVE-BLOCK-END
