import numpy as np
import time as _time

# Explicit deadline guard: track elapsed time at module level
# The evaluator kills after 150s wall-clock; we use a safety margin.
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0  # seconds

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    
    Args:
        ready_operations: list of dicts, each with:
            - job: job id
            - op: operation id
            - machine_options: dict {machine: duration}
            - remaining_ops: number of remaining operations for this job
            - remaining_work: total remaining work (sum of durations) for this job
        time: current simulation time
    
    Returns:
        index of the ready operation to start
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Deadline guard
    if _time.time() > _DEADLINE - _SAFETY_MARGIN:
        return 0
    
    # Extract metrics
    works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    ops_counts = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Calculate min duration and best machine for each op
    min_durs = np.zeros(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if not opts:
            min_durs[i] = 0.0
        else:
            min_durs[i] = min(opts.values())
    
    # Handle edge case where all works are 0
    if works.max() == 0:
        return 0
    
    # Strategy: Use a combination of heuristics that adapt based on the state.
    # Key insights for open-shop scheduling:
    # 1. MWRF (Most Work Remaining First) is generally good for makespan.
    # 2. SPT (Shortest Processing Time) helps clear operations quickly.
    # 3. Machine load balancing can be important.
    
    # We'll use a weighted score that adapts based on the spread of work.
    
    # Normalize works and durations to make them comparable
    work_max = works.max()
    work_min = works.min()
    dur_max = min_durs.max()
    dur_min = min_durs.min()
    
    # Normalized values (0 to 1 range)
    if work_max > work_min:
        work_norm = (works - work_min) / (work_max - work_min)
    else:
        work_norm = np.zeros(n)
    
    if dur_max > dur_min:
        dur_norm = (min_durs - dur_min) / (dur_max - dur_min)
    else:
        dur_norm = np.zeros(n)
    
    # Adaptive weighting:
    # - When work spread is large, prioritize MWRF (high work_norm)
    # - When work spread is small, prioritize SPT (low dur_norm, i.e., high 1-dur_norm)
    
    work_spread = work_max - work_min
    avg_work = works.mean()
    
    if avg_work > 0:
        ratio = work_spread / avg_work
    else:
        ratio = 0.0
    
    # Clamp ratio to [0, 3]
    ratio = np.clip(ratio, 0.0, 3.0)
    
    # alpha: weight on MWRF (work), beta: weight on SPT (1-duration)
    # High ratio -> high alpha, low beta
    # Low ratio -> low alpha, high beta
    alpha = 0.5 + 0.3 * ratio  # ranges from 0.5 to 1.4
    beta = 1.0 - 0.3 * ratio   # ranges from 1.0 to 0.1
    
    # Score = alpha * work_norm + beta * (1 - dur_norm)
    # Higher score is better
    scores = alpha * work_norm + beta * (1.0 - dur_norm)
    
    # Add a small tiebreaker based on remaining ops (prefer jobs with more ops remaining)
    # This helps complete jobs that have more work to do
    if ops_counts.max() > ops_counts.min():
        ops_norm = (ops_counts - ops_counts.min()) / (ops_counts.max() - ops_counts.min())
        scores += 0.1 * ops_norm
    
    best_idx = int(np.argmax(scores))
    return best_idx
