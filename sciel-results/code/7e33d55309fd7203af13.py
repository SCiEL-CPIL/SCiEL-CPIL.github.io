"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 145.0  # safety margin before 150s kill
_START_TIME = _time.time()


def _elapsed():
    return _time.time() - _START_TIME


def dispatch(ready_operations, time):
    """
    Greedy Bottleneck Elimination (LPT-style) with ratio-based tie-breaking.
    
    Strategy:
    1. Primary: pick the operation with the maximum minimum processing time
       (i.e., the operation that has the longest unavoidable processing time
       across its machine options). This is LPT — schedule long, inflexible
       operations early to prevent them from becoming critical-path bottlenecks.
    2. Secondary: among ties, prefer the job with more remaining work (urgency).
    3. Tertiary: prefer the job with more remaining operations (to keep pipeline full).
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # For each ready operation, compute:
    #   min_duration: the minimum duration across its machine options (the "bottleneck" length)
    #   remaining_work: total remaining work for that job
    #   remaining_ops: number of remaining operations for that job
    
    min_durations = np.empty(n, dtype=np.float64)
    remaining_works = np.empty(n, dtype=np.float64)
    remaining_ops_arr = np.empty(n, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        # Minimum duration across available machines
        if opts:
            min_durations[i] = min(opts.values())
        else:
            min_durations[i] = 0.0
        remaining_works[i] = op.get("remaining_work", 0.0)
        remaining_ops_arr[i] = op.get("remaining_ops", 0)
    
    # Primary key: max min_duration (LPT — schedule the longest bottleneck first)
    # Secondary key: max remaining_work (urgency among ties)
    # Tertiary key: max remaining_ops (keep pipeline full)
    
    # Combine into a single score for argmax:
    # We want to maximize min_duration primarily, then remaining_work, then remaining_ops.
    # Use a lexicographic approach via sorting.
    
    # Normalize to avoid scale issues
    max_md = np.max(min_durations) if n > 0 else 1.0
    max_rw = np.max(remaining_works) if n > 0 else 1.0
    max_ro = np.max(remaining_ops_arr) if n > 0 else 1.0
    
    if max_md == 0:
        max_md = 1.0
    if max_rw == 0:
        max_rw = 1.0
    if max_ro == 0:
        max_ro = 1.0
    
    # Lexicographic: primary = min_duration/max_md, secondary = remaining_work/max_rw, tertiary = remaining_ops/max_ro
    # Weight heavily toward primary
    scores = (min_durations / max_md) * 1.0 + (remaining_works / max_rw) * 0.1 + (remaining_ops_arr / max_ro) * 0.01
    
    idx = int(np.argmax(scores))
    return idx
# EVOLVE-BLOCK-END
