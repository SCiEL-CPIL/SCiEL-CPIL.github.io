"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.monotonic()
_DEADLINE = _START + 145.0  # 5s safety margin

def _deadline_exceeded():
    return _time.monotonic() > _DEADLINE

def _normalize(arr):
    """Normalize array to [0, 1] range."""
    arr = np.asarray(arr, dtype=np.float64)
    mn = arr.min()
    mx = arr.max()
    if mx == mn:
        return np.ones_like(arr) * 0.5
    return (arr - mn) / (mx - mn)

def _soft_min(values, alpha=2.0):
    """Compute soft-min using log-sum-exp trick for smooth approximation."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    # Use negative log-sum-exp of negative values
    neg_vals = -values
    max_neg = np.max(neg_vals)
    shifted = neg_vals - max_neg
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    if sum_exp == 0:
        return 0.0
    # Soft min = -log(sum(exp(-x))) = -(max_neg + log(sum(exp(x - max_neg))))
    log_sum = max_neg + np.log(sum_exp)
    return -log_sum

def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    Returns index of the ready operation to start next.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Check deadline
    if _deadline_exceeded():
        return 0

    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # For each op, collect all machine durations
    all_durations = []
    for op in ready_operations:
        if op["machine_options"]:
            all_durations.append(list(op["machine_options"].values()))
        else:
            all_durations.append([1.0])
    
    # Compute min, mean, and soft-min durations
    min_durations = np.array([min(d) if d else 1.0 for d in all_durations], dtype=np.float64)
    mean_durations = np.array([np.mean(d) if d else 1.0 for d in all_durations], dtype=np.float64)
    
    # Use soft-min for smooth approximation
    soft_min_durations = np.array([_soft_min(np.array(d, dtype=np.float64)) if d else 1.0 
                                    for d in all_durations], dtype=np.float64)

    # Normalize features
    nw = _normalize(remaining_work)
    no = _normalize(remaining_ops)
    nd_min = _normalize(min_durations)
    nd_mean = _normalize(mean_durations)
    nd_soft = _normalize(soft_min_durations)

    # Compute average work per remaining op
    avg_work_per_op = remaining_work / np.maximum(remaining_ops, 1.0)
    naw = _normalize(avg_work_per_op)

    # Dynamic bottleneck identification:
    # For each ready operation, estimate its "slack" based on current time and 
    # the shortest possible completion. Operations with less slack (more urgent) 
    # should be prioritized to prevent machine starvation.
    
    # Estimate completion time for each op if started now
    # Using soft-min duration as a smooth proxy for the actual processing time
    est_completion = time + soft_min_durations
    
    # Slack: how much buffer does this op have before it becomes critical?
    # We don't know the true due date, so use remaining work as a proxy for urgency.
    # Ops with high remaining work relative to their estimated duration are more critical.
    
    # Criticality score: high remaining work + low estimated duration = more critical
    criticality = nw * (1.0 - nd_soft)
    
    # Weights: balance urgency, flow, SPT, work-per-op, and criticality
    w_work = 0.25      # urgency (higher remaining work)
    w_ops = 0.15       # keep in motion (higher remaining ops)
    w_spt = 0.25       # shortest processing time (lower duration is better)
    w_avg = 0.10       # average work per op
    w_crit = 0.25      # criticality (high work, short duration)
    
    # Score: maximize
    score = (w_work * nw + 
             w_ops * no + 
             w_spt * (1.0 - nd_soft) + 
             w_avg * (1.0 - naw) +
             w_crit * criticality)

    # Pick the operation with the highest score
    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
