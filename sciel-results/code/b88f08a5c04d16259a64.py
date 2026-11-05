"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = None

def _ensure_init():
    global _dispatch_start_time
    if _dispatch_start_time is None:
        _dispatch_start_time = _time.time()

def _time_left():
    if _dispatch_start_time is None:
        return 115.0
    return 115.0 - (_time.time() - _dispatch_start_time)

def dispatch(ready_operations, time):
    _ensure_init()
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # If we're running low on time, use the fastest heuristic
    if _time_left() < 5.0:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Compute scores for each ready operation
    # Strategy: Combine multiple heuristics for open-shop scheduling
    # Key insight: In open-shop, we want to minimize makespan.
    # The critical path is determined by the longest job (sum of all ops) and 
    # the busiest machine (sum of all ops on that machine).
    # A good heuristic is to prioritize operations that are on the critical path.
    
    scores = []
    for op in ready_operations:
        work = op["remaining_work"]
        durations = list(op["machine_options"].values())
        if not durations:
            scores.append(0.0)
            continue
        
        min_dur = min(durations)
        max_dur = max(durations)
        avg_dur = sum(durations) / len(durations)
        num_machines = len(durations)
        remaining_ops = op.get("remaining_ops", 0)
        
        # Primary: Most Work Remaining (MWR)
        # This is the dominant factor in open-shop scheduling
        score = work
        
        # Secondary: Prefer operations with shorter processing times
        # to keep machines busy and reduce idle time
        # Use min duration as a proxy for "quick to complete"
        score += 0.1 * (1.0 / (min_dur + 1e-9))
        
        # Tertiary: Prefer operations that are closer to completing their job
        # Completing a job frees it from consideration
        if remaining_ops > 0:
            score += 0.05 * (1.0 / remaining_ops)
        
        # Quaternary: Slight penalty for very long operations
        # to avoid blocking a machine for too long
        score -= 0.02 * max_dur
        
        # Quinary: Prefer operations with more machine options (flexibility)
        score += 0.01 * num_machines
        
        # Senary: Smooth surrogate for min duration using soft-min (log-sum-exp trick)
        # This provides a smoother gradient than hard min, helping with tie-breaking
        # and potentially finding better assignments when durations are close
        alpha = 0.5  # Temperature parameter for soft-min
        # soft_min = -log(sum(exp(-alpha * d))) / alpha
        # For numerical stability, we use the fact that soft_min is between min and max
        # and approaches min as alpha -> inf
        # Here we use a simpler approximation: weighted geometric mean
        # which is a smooth surrogate for min
        if num_machines > 1:
            # Geometric mean as smooth surrogate for min
            log_sum = sum(np.log(d + 1e-9) for d in durations)
            geo_mean = np.exp(log_sum / num_machines)
            # Blend: use geo_mean instead of min_dur for smoother behavior
            score += 0.05 * (1.0 / (geo_mean + 1e-9))
        
        scores.append(score)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
