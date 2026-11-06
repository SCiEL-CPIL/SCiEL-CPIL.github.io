"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def dispatch(ready_operations, time):
    # Composite heuristic: balance remaining work, shortest processing time,
    # and job completion proximity to minimize makespan in open-shop scheduling.
    #
    # The idea:
    # 1. Prefer operations with high remaining work (MW) to keep busy machines.
    # 2. Among those, prefer shorter durations to reduce blocking.
    # 3. Consider how close the job is to completion to prioritize finishing jobs.
    #
    # We compute a composite score for each ready operation and pick the max.
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Precompute arrays for speed
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # For each operation, find the minimum duration across its machine options
    # This represents the best-case processing time if we could choose the fastest machine
    min_durations = np.zeros(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            min_durations[i] = min(mo.values())
        else:
            min_durations[i] = 0.0
    
    # Normalize each component to [0, 1] range for fair weighting
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr)
        return (arr - mn) / (mx - mn)
    
    rw_norm = normalize(remaining_works)
    ro_norm = normalize(remaining_ops)
    
    # For min_durations, we want to prefer shorter durations, so invert
    # normalize so that shorter = higher score
    md_norm = 1.0 - normalize(min_durations)
    
    # Job completion proximity: jobs with fewer remaining ops are closer to done
    # This is already captured by ro_norm (higher when fewer ops remain)
    
    # Composite score: weight remaining work most heavily, then shortest duration,
    # then job completion proximity
    # Tuned weights: MW=0.5, SPT=0.3, JCP=0.2
    score = 0.5 * rw_norm + 0.3 * md_norm + 0.2 * ro_norm
    
    # Pick the operation with the highest composite score
    best_idx = int(np.argmax(score))
    
    return best_idx
# EVOLVE-BLOCK-END
