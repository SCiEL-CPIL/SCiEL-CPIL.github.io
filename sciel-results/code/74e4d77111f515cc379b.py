"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.monotonic()
_DEADLINE = _START + 145.0  # 5s safety margin before 150s kill

def _deadline_exceeded():
    return _time.monotonic() >= _DEADLINE

def _normalize(arr):
    """Normalize array to [0, 1] range."""
    arr = np.asarray(arr, dtype=np.float64)
    mn = arr.min()
    mx = arr.max()
    if mx == mn:
        return np.ones_like(arr) * 0.5
    return (arr - mn) / (mx - mn)

def _soft_min(values, alpha=3.0):
    """Compute soft-min using log-sum-exp trick for smooth approximation of min."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 1.0
    if len(values) == 1:
        return values[0]
    neg_alpha_vals = -alpha * values
    max_neg = np.max(neg_alpha_vals)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_alpha_vals - max_neg)))
    return -log_sum / alpha

def _soft_max(values, alpha=3.0):
    """Compute soft-max using log-sum-exp trick for smooth approximation of max."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return values[0]
    alpha_vals = alpha * values
    max_alpha = np.max(alpha_vals)
    log_sum = max_alpha + np.log(np.sum(np.exp(alpha_vals - max_alpha)))
    return log_sum / alpha

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
    
    # For each op, collect machine durations and machine keys
    all_durations = []
    all_machine_keys = []
    for op in ready_operations:
        if op["machine_options"]:
            all_durations.append(list(op["machine_options"].values()))
            all_machine_keys.append(list(op["machine_options"].keys()))
        else:
            all_durations.append([1.0])
            all_machine_keys.append([0])
    
    # Compute min, max, mean durations
    min_durations = np.array([min(d) if d else 1.0 for d in all_durations], dtype=np.float64)
    max_durations = np.array([max(d) if d else 1.0 for d in all_durations], dtype=np.float64)
    mean_durations = np.array([np.mean(d) if d else 1.0 for d in all_durations], dtype=np.float64)
    
    # Soft-min for smooth duration estimation
    soft_min_durations = np.array([
        _soft_min(np.array(d, dtype=np.float64), alpha=3.0) if d else 1.0 
        for d in all_durations
    ], dtype=np.float64)

    # Estimate machine load: use remaining_work as proxy for how busy machines are
    # For each operation, the "load" of its preferred machine is estimated by
    # how much remaining work is already assigned to that machine across all ops
    # Since we don't track machine state, we use a heuristic: operations with
    # shorter min_duration are less likely to block, so their machine is "less loaded"
    
    # Dynamic bottleneck identification:
    # For each candidate op i, simulate assigning it to its least-loaded machine.
    # The "least-loaded" machine for op i is the one with the smallest duration
    # (since shorter duration means less time the machine is occupied).
    # After assignment, compute the slack for each other ready op j:
    #   slack_j = remaining_work_j - estimated_completion_time_j
    # We approximate: if op i takes duration d_i on its machine, it delays
    # other ops that might need the same machine.
    # 
    # Simpler approach: For each op i, compute the "bottleneck impact" as
    # the minimum over all ops j of (remaining_work_j - soft_min_duration_j)
    # after hypothetically removing op i's contribution to machine contention.
    #
    # Actually, let's use a cleaner formulation:
    # For each op i, compute the slack vector where:
    #   slack_j = remaining_work_j / soft_min_duration_j  (WSPT-like urgency)
    # The minimum slack across all ops indicates the most starved op.
    # We want to pick op i such that the minimum slack is maximized.
    #
    # But we need to account for the fact that assigning op i will change
    # the effective slack of other ops. Approximate: assigning op i with
    # duration d_i reduces the available time for other ops by d_i (if they
    # share machines). Since we don't know machine sharing, use a soft penalty.
    
    # Compute baseline WSPT for each op
    wspt = remaining_work / np.maximum(soft_min_durations, 1e-9)
    
    # For each candidate op i, compute the expected minimum slack after assignment
    # The idea: if we start op i, it will occupy a machine for ~soft_min_duration[i].
    # Other ops that might need that machine will have reduced slack.
    # Approximate the impact as: for each op j, effective_slack_j = wspt_j - penalty(i,j)
    # where penalty(i,j) is higher if op i and op j likely share machines.
    #
    # Simpler: use the min of wspt as a proxy for the most starved op.
    # When we pick op i, the new min slack is approximately:
    #   min_j (wspt_j) for j != i, but adjusted for the fact that op i is now in progress.
    #
    # Let's use: score_i = min_{j != i} wspt_j  (the minimum slack among remaining ops)
    # We want to maximize this, so we pick the op whose removal leaves the highest min slack.
    #
    # This is equivalent to: pick the op i that is NOT the bottleneck (has highest wspt),
    # so that the remaining ops have better minimum slack.
    
    # Compute min slack if we pick each op i
    # For n ops, this is O(n^2) which is fine for small n
    slack_scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if n == 1:
            slack_scores[i] = wspt[i]
        else:
            # Min slack among all ops except i
            other_wspt = np.delete(wspt, i)
            # Use soft-min for smoothness
            slack_scores[i] = _soft_min(other_wspt, alpha=3.0)
    
    # Normalize slack scores
    n_slack = _normalize(slack_scores)
    
    # Also compute the original multi-criteria score
    n_wspt = _normalize(wspt)
    nw = _normalize(remaining_work)
    no = _normalize(remaining_ops)
    nd_min = _normalize(min_durations)
    nd_max = _normalize(max_durations)
    nd_mean = _normalize(mean_durations)
    nd_soft = _normalize(soft_min_durations)
    
    # Original multi-criteria score
    w_wspt = 0.30
    w_work = 0.15
    w_ops = 0.10
    w_spt = 0.20
    w_range = 0.05
    
    original_score = (w_wspt * n_wspt + 
                      w_work * nw + 
                      w_ops * no + 
                      w_spt * (1.0 - nd_soft) + 
                      w_range * (1.0 - nd_max))
    
    # Combine: emphasize slack maximization (bottleneck prevention) with urgency
    w_slack = 0.40   # primary: prevent machine starvation
    w_orig = 0.60    # secondary: original urgency/flow signals
    
    score = w_slack * n_slack + w_orig * original_score
    
    # Pick the operation with the highest score
    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
