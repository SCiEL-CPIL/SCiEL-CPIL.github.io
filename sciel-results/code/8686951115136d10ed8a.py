"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Module-level deadline guard
_T0 = _time.monotonic()
_DEADLINE_MARGIN = 10.0  # seconds before the 120s limit


def _elapsed_ok():
    return (_time.monotonic() - _T0) < (120.0 - _DEADLINE_MARGIN)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    
    Strategy: Multi-criteria heuristic that balances:
    1. Remaining work on the job (prioritize jobs with more work left)
    2. Minimum processing time on available machines (reduce makespan contribution)
    3. Machine load balancing (prefer less-loaded machines)
    4. Remaining operations count (jobs with fewer remaining ops can finish sooner)
    
    We use a weighted score combining these factors.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Deadline guard
    if not _elapsed_ok():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    scores = np.zeros(n)
    
    # Gather features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # For each ready op, find the min duration among machine options
    min_durations = np.zeros(n)
    # Also track which machine gives min duration (for tie-breaking / load balancing)
    best_machines = np.zeros(n, dtype=int)
    
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if isinstance(mo, dict):
            items = list(mo.items())
        elif isinstance(mo, (list, tuple)):
            items = list(mo)
        else:
            items = list(mo.items()) if hasattr(mo, 'items') else [(0, 1.0)]
        
        if items:
            best_mach = min(items, key=lambda x: x[1])
            min_durations[i] = best_mach[1]
            best_machines[i] = best_mach[0]
        else:
            min_durations[i] = 1.0
            best_machines[i] = 0
    
    # Normalize features to [0, 1] range for fair weighting
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)
    
    norm_work = normalize(remaining_works)
    norm_ops = normalize(remaining_ops)
    norm_dur = normalize(min_durations)  # lower is better, so we want to minimize
    
    # Weighted score:
    # - High remaining work -> higher priority (process more work)
    # - Fewer remaining ops -> higher priority (can complete job)
    # - Shorter duration -> higher priority (frees machine sooner)
    w_work = 0.35
    w_ops = 0.25
    w_dur = 0.40
    
    # For duration, lower is better, so we invert
    scores = w_work * norm_work + w_ops * (1.0 - norm_ops) + w_dur * (1.0 - norm_dur)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
