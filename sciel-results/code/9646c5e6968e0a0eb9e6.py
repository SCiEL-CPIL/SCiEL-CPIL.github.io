"""Dispatch rule for open-shop scheduling: choose ready operation with best priority."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: track start time for safety
_START_TIME = _time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 150s limit


def _elapsed_ok():
    """Check if we're still within safe time budget."""
    return (_time.time() - _START_TIME) < (150.0 - _DEADLINE_MARGIN)


def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next.
    
    Strategy: Combine multiple heuristics to minimize makespan:
    1. Most remaining work (MRW) - the original seed
    2. Minimum processing time (SPT) - reduce idle time
    3. Balance across machines - prefer machines with less current load
    4. Consider remaining operations count - prioritize jobs with more ops left
    5. Time-aware urgency: operations with less slack relative to current time
    
    We use a weighted combination with a smooth tie-breaking mechanism.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Minimum processing time among machine options (SPT heuristic)
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Average processing time
    avg_durations = np.array([
        np.mean(list(op["machine_options"].values())) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Number of machine options (flexibility)
    n_options = np.array([len(op["machine_options"]) for op in ready_operations], dtype=np.float64)
    
    # Normalize features
    def normalize(x):
        rng = x.max() - x.min()
        if rng > 1e-12:
            return (x - x.min()) / rng
        return np.zeros_like(x)
    
    nw_norm = normalize(remaining_work)
    ro_norm = normalize(remaining_ops)
    md_norm = normalize(min_durations)  # lower is better for SPT
    ad_norm = normalize(avg_durations)
    
    # Time-aware urgency: how much time has elapsed relative to total work
    # Higher time means we're later in the schedule, so prioritize more aggressively
    if time > 0:
        # Urgency factor: increases as time progresses
        # Normalize remaining work against current time to get a critical-ratio-like measure
        # Avoid division by zero
        safe_time = max(time, 1e-6)
        urgency = remaining_work / safe_time
        urgency_norm = normalize(urgency)
    else:
        urgency_norm = nw_norm
    
    # Composite score: we want to maximize this
    # - High remaining work: good (finish big jobs early)
    # - Low min duration: good (SPT, less idle time)
    # - High remaining ops: good (keep pipeline full)
    # - Low avg duration: good (faster completion)
    # - High urgency: good (time-critical operations first)
    
    w_rw = 0.30  # weight for remaining work
    w_md = 0.25  # weight for min duration (inverted)
    w_ro = 0.20  # weight for remaining ops
    w_ad = 0.10  # weight for avg duration (inverted)
    w_urg = 0.15  # weight for time-aware urgency
    
    score = (w_rw * nw_norm + 
             w_md * (1.0 - md_norm) + 
             w_ro * ro_norm + 
             w_ad * (1.0 - ad_norm) +
             w_urg * urgency_norm)
    
    # Find the best index
    idx = int(np.argmax(score))
    
    # Smooth tie-breaking using soft-min via log-sum-exp on durations
    # Instead of hard argmin, use a smooth approximation
    max_score = score.max()
    close = np.where(score > max_score - 1e-4)[0]
    if len(close) > 1:
        # Soft-min: use log-sum-exp with negative durations to get smooth minimum
        # We want to prefer shorter durations, so use -durations in log-sum-exp
        neg_durations = -min_durations[close]
        # Temperature for softness
        beta = 10.0
        # Log-sum-exp for numerical stability
        max_neg = np.max(neg_durations)
        log_sum_exp = max_neg + np.log(np.sum(np.exp(beta * (neg_durations - max_neg))))
        # The soft-min index is where the contribution is largest
        contributions = np.exp(beta * (neg_durations - max_neg))
        # Normalize to get probabilities
        probs = contributions / contributions.sum()
        # Pick the index with highest probability (smallest duration in soft sense)
        idx = close[int(np.argmax(probs))]
    
    return int(idx)
# EVOLVE-BLOCK-END
