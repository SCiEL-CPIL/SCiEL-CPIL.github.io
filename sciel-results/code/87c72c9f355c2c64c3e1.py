"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Global deadline guard
_START_TIME = _time.time()
_HARD_LIMIT = 110.0  # seconds
_SOFT_MARGIN = 5.0   # seconds

def _check_deadline():
    """Return True if we are approaching the hard limit."""
    return (_time.time() - _START_TIME) > (_HARD_LIMIT - _SOFT_MARGIN)

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Bipartite graph-inspired heuristic. For each ready operation,
    compute the maximum over machines of (remaining_work / processing_time_on_machine).
    This prioritizes operations that yield the greatest reduction in critical
    path per unit of machine time. We also blend in LPT (longest processing time)
    to keep machines busy with long tasks.
    
    The score for operation i is:
        score_i = alpha * max_m (remaining_work_i / p_{i,m}) + (1-alpha) * norm_max_duration_i
    
    where p_{i,m} is the processing time of operation i on machine m,
    and norm_max_duration_i is the normalized maximum duration across machines.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Check deadline - if too close, fall back to a very fast rule
    if _check_deadline():
        # Fallback: pick the one with max remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Compute for each operation:
    # 1. max_ratio: max over machines of (remaining_work / duration)
    # 2. max_duration: max over machines of duration (for LPT component)
    
    max_ratios = np.zeros(n)
    max_durations = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        rw = op["remaining_work"]
        if machine_options:
            durations = list(machine_options.values())
            max_durations[i] = max(durations)
            # Compute ratio for each machine, take max
            # Guard against zero duration
            ratios = []
            for d in durations:
                if d > 0:
                    ratios.append(rw / d)
            if ratios:
                max_ratios[i] = max(ratios)
            else:
                max_ratios[i] = 0.0
        else:
            max_durations[i] = 0.0
            max_ratios[i] = 0.0
    
    # Normalize both criteria to [0, 1]
    ratio_range = np.ptp(max_ratios)
    dur_range = np.ptp(max_durations)
    
    if ratio_range > 0:
        norm_ratios = (max_ratios - max_ratios.min()) / ratio_range
    else:
        norm_ratios = np.zeros(n)
    
    if dur_range > 0:
        norm_durations = (max_durations - max_durations.min()) / dur_range
    else:
        norm_durations = np.zeros(n)
    
    # Weighted composite: 60% ratio (critical path reduction), 40% LPT
    alpha = 0.6
    composite_scores = alpha * norm_ratios + (1 - alpha) * norm_durations
    
    best_idx = int(np.argmax(composite_scores))
    
    return best_idx
# EVOLVE-BLOCK-END
