"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling using a normalized weighted sum heuristic.
    
    Combines:
    - Most Work Remaining (MWR): Primary indicator of critical path.
    - Shortest Processing Time (SPT): Secondary indicator to free machines.
    - Remaining Operations (RO): Tertiary indicator to complete jobs.
    
    Normalization ensures that features with different scales (e.g., work units vs inverse time)
    contribute proportionally to the score, avoiding the dominance of one feature over others
    which can occur in product-based heuristics when features are negatively correlated.
    """
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _time.time() - _START_TIME > _TIME_LIMIT:
        return 0
    
    n = len(ready_operations)
    
    # If only one operation, no choice
    if n == 1:
        return 0
    
    # Extract features for each ready operation
    mwr_list = []
    spt_list = []
    ro_list = []
    
    # Cache MWR per job to avoid redundant lookups if multiple ops of same job are ready
    job_mwr_cache = {}
    
    for op in ready_operations:
        job = op["job"]
        if job not in job_mwr_cache:
            job_mwr_cache[job] = op["remaining_work"]
        
        # MWR: Most Work Remaining for the job
        mwr_list.append(job_mwr_cache[job])
        
        # SPT: Inverse of minimum duration (higher is better/faster)
        durations = list(op["machine_options"].values())
        min_dur = min(durations) if durations else 1.0
        spt_list.append(1.0 / (min_dur + 1e-9))
        
        # RO: Inverse of remaining operations count (higher is better/closer to finish)
        ro_list.append(1.0 / (op["remaining_ops"] + 1e-9))
    
    # Convert to numpy arrays for vectorized normalization
    mwr_arr = np.array(mwr_list)
    spt_arr = np.array(spt_list)
    ro_arr = np.array(ro_list)
    
    # Helper to normalize an array to [0, 1]
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        range_val = mx - mn
        if range_val < 1e-9:
            return np.ones_like(arr)
        return (arr - mn) / range_val
    
    n_mwr = normalize(mwr_arr)
    n_spt = normalize(spt_arr)
    n_ro = normalize(ro_arr)
    
    # Weighted Sum
    # MWR is the strongest predictor of makespan in JSS.
    # SPT helps reduce idle time.
    # RO helps close out jobs, reducing tail latency.
    
    # Weights tuned to prioritize MWR heavily, but allow SPT and RO to break ties
    # or provide secondary guidance.
    w_mwr = 0.6
    w_spt = 0.3
    w_ro = 0.1
    
    scores = w_mwr * n_mwr + w_spt * n_spt + w_ro * n_ro
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
