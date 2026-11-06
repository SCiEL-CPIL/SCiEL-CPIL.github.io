"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 120.0
_START_TIME = None

def _ensure_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.monotonic()

def _elapsed():
    _ensure_start()
    return _time.monotonic() - _START_TIME

def _softmax(x, temperature=1.0):
    """Stable softmax for ranking."""
    if len(x) == 0:
        return np.array([])
    x_max = np.max(x)
    exp_x = np.exp((x - x_max) / temperature)
    return exp_x / np.sum(exp_x)

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    
    Strategy: Composite heuristic with critical-path lookahead and soft-max smoothing.
    
    Features:
    1. remaining_work: more is better (long jobs should progress)
    2. min_duration: shorter is better (frees machines quickly)
    3. remaining_ops: fewer is better (jobs closer to completion)
    4. critical_path_estimate: estimated time until job completion if prioritized
    
    We use a weighted combination with soft-max to smooth the ranking.
    """
    _ensure_start()
    
    if not ready_operations:
        return 0
    
    # Check deadline with safety margin
    if _elapsed() > _DEADLINE - 5.0:
        # Fallback to simple rule if we're running out of time
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n = len(ready_operations)
    
    # Compute features for each ready operation
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    
    # For each operation, find min and avg duration among its machine options
    min_durations = np.zeros(n, dtype=float)
    avg_durations = np.zeros(n, dtype=float)
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if machine_options:
            durations = list(machine_options.values())
            min_durations[i] = min(durations)
            avg_durations[i] = np.mean(durations)
        else:
            min_durations[i] = 0.0
            avg_durations[i] = 0.0
    
    # Critical path estimate: if we prioritize this job, when will it complete?
    # Approximation: remaining_work / (num_machines * 0.5) as a proxy for parallelism
    # Jobs with more remaining work and more remaining ops are on the critical path
    num_machines = 5  # Fixed by task
    critical_path = remaining_works / (num_machines * 0.5)
    
    # Normalize features to [0, 1] range
    def normalize(arr, invert=False):
        if np.max(arr) > np.min(arr):
            norm = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
            return 1.0 - norm if invert else norm
        return np.ones_like(arr) if invert else np.zeros_like(arr)
    
    # remaining_work: higher is better
    rw_norm = normalize(remaining_works)
    
    # min_duration: lower is better (invert)
    md_norm = normalize(min_durations, invert=True)
    
    # remaining_ops: lower is better (invert)  
    ro_norm = normalize(remaining_ops, invert=True)
    
    # critical_path: higher is better (we want to prioritize critical path jobs)
    cp_norm = normalize(critical_path)
    
    # Weighted combination with emphasis on critical path and remaining work
    # Weights tuned to balance urgency and machine utilization
    scores = 0.35 * rw_norm + 0.25 * cp_norm + 0.25 * md_norm + 0.15 * ro_norm
    
    # Use soft-max to smooth the ranking (temperature controls sensitivity)
    # Lower temperature = more confident in top choice; higher = more balanced
    temperature = 0.5
    probs = _softmax(scores, temperature)
    
    # Pick the operation with the highest probability
    return int(np.argmax(probs))
# EVOLVE-BLOCK-END
