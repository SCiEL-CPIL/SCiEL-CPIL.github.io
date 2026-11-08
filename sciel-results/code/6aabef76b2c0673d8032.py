"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Global deadline guard to ensure we finish well within the 120s evaluation limit
_START_TIME = _time.time()
_HARD_LIMIT = 110.0  # seconds
_SOFT_MARGIN = 5.0   # seconds buffer before hard limit


def _check_deadline():
    """Return True if we are approaching the hard limit."""
    return (_time.time() - _START_TIME) > (_HARD_LIMIT - _SOFT_MARGIN)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    Uses a hybrid heuristic combining multiple scheduling rules to minimize makespan.
    """
    if not ready_operations:
        return 0
    
    # Check if we're approaching the deadline - if so, use a fast fallback
    if _check_deadline():
        # Fast fallback: pick the operation with the most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Calculate machine loads from machine_options
    # For each ready operation, find the minimum duration machine option
    min_durations = []
    for op in ready_operations:
        machine_options = op["machine_options"]
        if machine_options:
            min_dur = min(machine_options.values())
        else:
            min_dur = 0
        min_durations.append(min_dur)
    
    # Extract features
    remaining_works = [op["remaining_work"] for op in ready_operations]
    remaining_ops_counts = [op["remaining_ops"] for op in ready_operations]
    
    # Heuristic 1: Most work remaining (MW)
    mw_scores = remaining_works
    
    # Heuristic 2: Longest processing time first (LPT) - use min duration
    lpt_scores = min_durations
    
    # Heuristic 3: Most operations remaining
    mor_scores = remaining_ops_counts
    
    # Combine heuristics with weights
    # Weighted combination: prioritize work remaining, then processing time, then ops count
    # Normalize scores to comparable scales
    max_mw = max(mw_scores) if max(mw_scores) > 0 else 1
    max_lpt = max(lpt_scores) if max(lpt_scores) > 0 else 1
    max_mor = max(mor_scores) if max(mor_scores) > 0 else 1
    
    normalized_mw = [x / max_mw for x in mw_scores]
    normalized_lpt = [x / max_lpt for x in lpt_scores]
    normalized_mor = [x / max_mor for x in mor_scores]
    
    # Combine with weights: MW (0.5), LPT (0.3), MOR (0.2)
    combined_scores = [
        0.5 * mw + 0.3 * lpt + 0.2 * mor
        for mw, lpt, mor in zip(normalized_mw, normalized_lpt, normalized_mor)
    ]
    
    # Return index of the operation with the highest combined score
    return int(np.argmax(combined_scores))
# EVOLVE-BLOCK-END
