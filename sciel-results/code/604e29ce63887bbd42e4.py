"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling using a product-based heuristic
    with soft-max smoothing to balance urgency and blocking penalties.
    
    Uses:
    - Most Work Remaining (MWR): Urgency indicator.
    - Shortest Processing Time (SPT): Blocking penalty (shorter = less blocking).
    - Remaining Operations (RO): Completion proximity.
    
    The product of normalized MWR and SPT prioritizes operations that are both
    urgent (high remaining work) and quick to complete (low blocking), which
    better identifies critical path reductions than additive weighting.
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
    
    # Product-based scoring with soft-max smoothing
    # The product n_mwr * n_spt captures the interaction between urgency and blocking.
    # Operations that are both high-urgency and low-blocking get disproportionately high scores.
    
    # Use a smooth product via log-sum-exp for numerical stability
    # score = exp(a * log(n_mwr + eps) + b * log(n_spt + eps) + c * log(n_ro + eps))
    # This is equivalent to (n_mwr)^a * (n_spt)^b * (n_ro)^c
    
    # Exponents tuned to emphasize MWR-SPT interaction while keeping RO as a tiebreaker
    a = 1.5  # MWR exponent
    b = 1.0  # SPT exponent
    c = 0.3  # RO exponent
    
    eps = 1e-9
    log_scores = a * np.log(n_mwr + eps) + b * np.log(n_spt + eps) + c * np.log(n_ro + eps)
    
    return int(np.argmax(log_scores))
# EVOLVE-BLOCK-END
