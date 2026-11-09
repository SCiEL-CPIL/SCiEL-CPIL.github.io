"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard to ensure we finish well before the 150s limit
# The evaluator kills after 150s, we use a safety margin.
_DEADLINE = 145.0
_START_TIME = _time.time()

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Check deadline
    if _time.time() - _START_TIME > _DEADLINE:
        return 0

    # Extract metrics
    works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    ops_counts = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
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

    # Normalize works to [0, 1]
    work_min = works.min()
    work_max = works.max()
    if work_max > work_min:
        norm_work = (works - work_min) / (work_max - work_min)
    else:
        norm_work = np.zeros(n)
        
    # Normalize durations to [0, 1]
    dur_min = min_durs.min()
    dur_max = min_durs.max()
    if dur_max > dur_min:
        norm_dur = (min_durs - dur_min) / (dur_max - dur_min)
    else:
        norm_dur = np.zeros(n)
        
    # Normalize ops counts to [0, 1]
    ops_min = ops_counts.min()
    ops_max = ops_counts.max()
    if ops_max > ops_min:
        norm_ops = (ops_counts - ops_min) / (ops_max - ops_min)
    else:
        norm_ops = np.zeros(n)

    # Calculate Coefficient of Variation for works
    mean_work = works.mean()
    if mean_work > 0:
        std_work = works.std()
        cv = std_work / mean_work
    else:
        cv = 0.0
        
    # Adaptive weights
    # If CV is high (large disparity in work), prioritize MWRF (norm_work)
    # If CV is low (similar work), prioritize SPT (1 - norm_dur)
    
    # Clamp CV to [0, 1] for stable weighting
    cv_clamped = np.clip(cv, 0.0, 1.0)
    
    # Weight for Work (MWRF)
    w_work = 0.5 + 0.5 * cv_clamped
    # Weight for SPT (Shortest Processing Time)
    # We want to maximize (1 - norm_dur), so the weight is applied to (1 - norm_dur)
    w_spt = 1.0 - w_work
    
    # Small weight for remaining ops to break ties and favor complex jobs
    w_ops = 0.05
    
    # Total score
    # Higher score is better
    scores = w_work * norm_work + w_spt * (1.0 - norm_dur) + w_ops * norm_ops
    
    best_idx = int(np.argmax(scores))
    return best_idx
# EVOLVE-BLOCK-END
