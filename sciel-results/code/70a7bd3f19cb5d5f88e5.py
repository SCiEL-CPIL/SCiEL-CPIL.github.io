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
    1. Most remaining work (MRW) - prioritize jobs with more work left
    2. Longest Processing Time (LPT) - prioritize operations with longer durations 
       to balance machine loads early and prevent late-stage bottlenecks.
    3. Remaining operations count - keep pipeline full
    4. Time-aware urgency: operations with less slack relative to current time
    
    We use a weighted combination with a robust tie-breaking mechanism.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Minimum processing time among machine options (used for LPT heuristic here)
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Average processing time
    avg_durations = np.array([
        np.mean(list(op["machine_options"].values())) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Normalize features
    def normalize(x):
        rng = x.max() - x.min()
        if rng > 1e-12:
            return (x - x.min()) / rng
        return np.zeros_like(x)
    
    nw_norm = normalize(remaining_work)
    ro_norm = normalize(remaining_ops)
    md_norm = normalize(min_durations)  # Higher is longer duration
    ad_norm = normalize(avg_durations)
    
    # Time-aware urgency: how much time has elapsed relative to total work
    if time > 0:
        safe_time = max(time, 1e-6)
        urgency = remaining_work / safe_time
        urgency_norm = normalize(urgency)
    else:
        urgency_norm = nw_norm
    
    # Composite score: we want to maximize this
    # Hypothesis: LPT (Longest Processing Time) helps balance loads early.
    # We prioritize long operations (high min_duration) and jobs with high remaining work.
    
    w_rw = 0.40  # weight for remaining work
    w_md = 0.35  # weight for min duration (positive: LPT)
    w_ro = 0.15  # weight for remaining ops
    w_urg = 0.10  # weight for time-aware urgency
    
    score = (w_rw * nw_norm + 
             w_md * md_norm + 
             w_ro * ro_norm +
             w_urg * urgency_norm)
    
    # Find the best index
    idx = int(np.argmax(score))
    
    # Robust tie-breaking:
    # If scores are very close, prefer the operation with the largest minimum duration (LPT consistency)
    # or the one with the most remaining work.
    max_score = score.max()
    close = np.where(score > max_score - 1e-6)[0]
    if len(close) > 1:
        # Among close candidates, pick the one with the longest minimum duration
        close_durations = min_durations[close]
        best_local = np.argmax(close_durations)
        idx = close[best_local]
    
    return int(idx)
# EVOLVE-BLOCK-END
