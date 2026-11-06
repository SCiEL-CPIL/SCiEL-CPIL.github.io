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


def _soft_min(durations, beta=5.0):
    """
    Smooth surrogate for min using log-sum-exp with negative sign.
    soft_min(x) = -log(sum(exp(-beta * x))) / beta
    As beta -> inf, this approaches true min.
    For finite beta, it's a smooth, differentiable approximation.
    """
    if len(durations) == 0:
        return 0.0
    if len(durations) == 1:
        return float(durations[0])
    x = -beta * durations
    # For numerical stability, subtract max before exp
    x_max = np.max(x)
    if x_max > 100:  # overflow guard
        return float(np.min(durations))
    exp_sum = np.sum(np.exp(x - x_max))
    if exp_sum <= 0 or not np.isfinite(exp_sum):
        return float(np.min(durations))
    return float(-np.log(exp_sum) / beta - x_max / beta)


def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next.
    
    Strategy (refined composite heuristic):
    1. Prioritize jobs with high remaining work (critical path).
    2. Among those, prefer operations with shorter effective duration using a
       smooth soft-min surrogate over machine options (captures both min duration
       and the "closeness" of alternatives).
    3. Prefer operations with more machine options (flexibility for load balancing).
    4. Assign the operation to the machine with the shortest processing time.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    
    soft_min_durations = np.zeros(n, dtype=np.float64)
    true_min_durations = np.zeros(n, dtype=np.float64)
    flexibilities = np.zeros(n, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if opts:
            durations = np.array(list(opts.values()), dtype=np.float64)
            soft_min_durations[i] = _soft_min(durations, beta=5.0)
            true_min_durations[i] = np.min(durations)
            flexibilities[i] = len(durations)
        else:
            soft_min_durations[i] = 0.0
            true_min_durations[i] = 0.0
            flexibilities[i] = 0.0
    
    # Normalize features
    def normalize(x):
        rng = x.max() - x.min()
        if rng > 1e-12:
            return (x - x.min()) / rng
        return np.zeros_like(x)
    
    rw_norm = normalize(remaining_work)
    sdm_norm = normalize(soft_min_durations)
    flex_norm = normalize(flexibilities)
    
    # Composite score
    # We want to maximize:
    # - Remaining work (job is critical)
    # - Flexibility (more options to balance load)
    # - Inverse of soft-min duration (shorter effective ops free machines sooner)
    #
    # The soft-min surrogate smooths the nonsmooth min, so operations whose
    # durations are all close to the minimum score slightly better than those
    # with one very short but others very long (which the hard min would treat
    # the same). This better reflects the true load-balancing benefit.
    
    # Weights:
    # Slightly increase weight on duration component since the soft-min is a
    # more informative signal than the hard min.
    
    w_rw = 0.55
    w_md = 0.35
    w_flex = 0.10
    
    # Invert soft_min_duration: shorter is better -> (1 - norm)
    score = (w_rw * rw_norm + 
             w_md * (1.0 - sdm_norm) + 
             w_flex * flex_norm)
    
    # Find the best index
    idx = int(np.argmax(score))
    
    # Tie-breaking: if scores are close, prefer higher remaining work, then shorter soft-min duration
    max_score = score.max()
    close = np.where(score > max_score - 1e-4)[0]
    if len(close) > 1:
        close_rw = remaining_work[close]
        close_sdm = soft_min_durations[close]
        
        def normalize_local(x):
            rng = x.max() - x.min()
            if rng > 1e-12:
                return (x - x.min()) / rng
            return np.zeros_like(x)
        
        # Prefer: high remaining work, LOW soft-min duration
        sec_score = (0.7 * normalize_local(close_rw) + 
                     0.3 * (1.0 - normalize_local(close_sdm)))
        best_local = int(np.argmax(sec_score))
        idx = int(close[best_local])
    
    # Select the best machine for the chosen operation
    # Strategy: Pick the machine with the shortest duration to minimize machine occupation time
    op = ready_operations[idx]
    machine_to_use = None
    min_dur = float('inf')
    
    for machine, duration in op["machine_options"].items():
        if duration < min_dur:
            min_dur = duration
            machine_to_use = machine
    
    # Return tuple (index, machine) to explicitly assign the machine
    # This allows us to control the assignment to the most efficient machine
    return (idx, machine_to_use)

# EVOLVE-BLOCK-END
