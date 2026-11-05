"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard
_DEADLINE_START = _time.time()
_DEADLINE_LIMIT = 140.0  # safety margin before 150s kill


def _check_deadline():
    return (_time.time() - _DEADLINE_START) > _DEADLINE_LIMIT


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # If only one option, return it
    if n == 1:
        return 0
    
    # Deadline check - if we're close to timeout, fall back to simple rule
    if _check_deadline():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    
    # Get minimum and average duration for each operation
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=float)
    
    avg_durations = np.array([
        np.mean(list(op["machine_options"].values())) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=float)
    
    # Count machine options (fewer options = more constrained)
    option_counts = np.array([
        len(op["machine_options"]) if op["machine_options"] else 0
        for op in ready_operations
    ], dtype=float)
    
    # Normalize features to [0, 1]
    def normalize(arr):
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val < 1e-9:
            return np.ones_like(arr) * 0.5
        return (arr - min_val) / (max_val - min_val)
    
    norm_work = normalize(remaining_works)
    norm_ops = normalize(remaining_ops)
    norm_dur = normalize(min_durations)
    norm_avg_dur = normalize(avg_durations)
    norm_options = normalize(option_counts)
    
    # Weights: emphasize remaining work, balance urgency and efficiency
    w_work = 0.45
    w_ops = 0.25
    w_dur = 0.15
    w_avg_dur = 0.10
    w_options = 0.05
    
    # Score: higher is better
    # More remaining work → higher priority
    # Fewer remaining ops → higher priority (invert)
    # Shorter min duration → higher priority (invert)
    # Shorter avg duration → higher priority (invert)
    # Fewer options → higher priority (more constrained, should be scheduled sooner)
    score = (w_work * norm_work + 
             w_ops * (1.0 - norm_ops) + 
             w_dur * (1.0 - norm_dur) +
             w_avg_dur * (1.0 - norm_avg_dur) +
             w_options * (1.0 - norm_options))
    
    # Add small tie-breaker based on job number for determinism
    job_ids = np.array([op["job"] for op in ready_operations], dtype=float)
    score += 1e-6 * (1.0 - normalize(job_ids))
    
    return int(np.argmax(score))
# EVOLVE-BLOCK-END
