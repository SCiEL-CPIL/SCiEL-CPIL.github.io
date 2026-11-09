"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# We keep a simple deadline guard. The dispatch call itself is cheap, but we
# track a global start to avoid pathological loops if the evaluator ever calls
# dispatch in a tight loop without advancing time.
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy:
    1. If only one op is ready, return it.
    2. Compute a score for each ready operation that balances:
       - remaining_work (total work left in the job)
       - minimum duration on any available machine (shortest processing time)
       - number of remaining ops (jobs with fewer remaining ops are more urgent)
    
    We use a composite heuristic:
    - Primary: minimize the minimum processing time across machine options (SPT-like)
    - Secondary: prioritize jobs with more remaining work (longest job first)
    - Tertiary: prioritize jobs with fewer remaining ops (to avoid starvation)
    
    This is a blend of SPT and LPT with urgency weighting.
    """
    # Deadline guard
    if _time.time() > _DEADLINE:
        return 0
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Extract features
    min_durs = []
    max_durs = []
    rem_works = []
    rem_ops = []
    
    for op in ready_operations:
        machine_options = op["machine_options"]
        durations = list(machine_options.values())
        min_durs.append(min(durations))
        max_durs.append(max(durations))
        rem_works.append(op["remaining_work"])
        rem_ops.append(op["remaining_ops"])
    
    min_durs = np.array(min_durs, dtype=np.float64)
    max_durs = np.array(max_durs, dtype=np.float64)
    rem_works = np.array(rem_works, dtype=np.float64)
    rem_ops = np.array(rem_ops, dtype=np.float64)
    
    # Normalize each feature to [0, 1]
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_min_dur = normalize(min_durs)
    norm_rem_work = normalize(rem_works)
    norm_rem_ops = normalize(rem_ops)
    
    # Score: we want to minimize this score (lower is better for selection)
    # - Lower min duration is better (SPT): weight 0.5
    # - Higher remaining work is better (LPT): weight 0.3 (so we subtract)
    # - Lower remaining ops is better (urgency): weight 0.2
    
    score = 0.5 * norm_min_dur - 0.3 * norm_rem_work + 0.2 * norm_rem_ops
    
    # Pick the operation with the lowest score
    idx = int(np.argmin(score))
    return idx
# EVOLVE-BLOCK-END
