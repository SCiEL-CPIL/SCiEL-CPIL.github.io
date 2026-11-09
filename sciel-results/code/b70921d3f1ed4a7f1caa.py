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

def _soft_min(values, alpha=2.0):
    """Compute soft-min using log-sum-exp trick for smooth approximation of min."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 1.0
    if len(values) == 1:
        return values[0]
    # Soft min: -1/alpha * log(sum(exp(-alpha * x)))
    neg_alpha_vals = -alpha * values
    max_neg = np.max(neg_alpha_vals)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_alpha_vals - max_neg)))
    return -log_sum / alpha

def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling using dynamic bottleneck identification.
    Selects the operation whose assignment to its least-loaded machine maximizes
    the minimum slack time across all currently ready operations.
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
    
    # For each op, collect machine durations
    all_durations = []
    for op in ready_operations:
        if op["machine_options"]:
            all_durations.append(list(op["machine_options"].values()))
        else:
            all_durations.append([1.0])
    
    # Compute min, max, mean, and soft-min durations
    min_durations = np.array([min(d) if d else 1.0 for d in all_durations], dtype=np.float64)
    max_durations = np.array([max(d) if d else 1.0 for d in all_durations], dtype=np.float64)
    mean_durations = np.array([np.mean(d) if d else 1.0 for d in all_durations], dtype=np.float64)
    
    # Soft-min for smooth duration estimation
    soft_min_durations = np.array([
        _soft_min(np.array(d, dtype=np.float64), alpha=3.0) if d else 1.0 
        for d in all_durations
    ], dtype=np.float64)

    # WSPT ratio: remaining_work / soft_min_duration
    wspt = remaining_work / np.maximum(soft_min_durations, 1e-9)
    
    # Normalize all features
    n_wspt = _normalize(wspt)
    nw = _normalize(remaining_work)
    no = _normalize(remaining_ops)
    nd_min = _normalize(min_durations)
    nd_max = _normalize(max_durations)
    nd_mean = _normalize(mean_durations)
    nd_soft = _normalize(soft_min_durations)

    # Dynamic bottleneck identification:
    # For each candidate operation i, estimate the "slack" it would provide
    # to other operations if scheduled now. The slack is approximated by how
    # much remaining work other ops have relative to their processing time,
    # minus the blocking effect of operation i on shared machines.
    
    # Compute a "blocking impact" matrix: for each pair (i, j), estimate how
    # much operation i delays operation j based on shared machine potential.
    # Since we don't have exact machine assignment, we use duration similarity
    # as a proxy for machine conflict likelihood.
    
    # For each candidate i, compute the minimum slack across all other ops j:
    # slack_j = remaining_work[j] / min_durations[j] - delay_from_i[j]
    # where delay_from_i[j] is an estimate of how much i delays j.
    
    # We approximate delay_from_i[j] as proportional to the overlap in
    # duration ranges between i and j (proxy for shared machine).
    
    # Compute duration overlap scores
    dur_overlap = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Overlap based on how similar the duration profiles are
            # Higher similarity -> more likely to share a machine -> more blocking
            overlap = 1.0 - abs(min_durations[i] - min_durations[j]) / (max(max_durations[i], max_durations[j]) + 1e-9)
            dur_overlap[i, j] = overlap
    
    # For each candidate i, compute the minimum slack it preserves
    min_slack_scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i == n - 1:
            min_slack_scores[i] = wspt[i]
            continue
        
        # Slacks for all other ops j != i
        slacks = np.zeros(n - 1, dtype=np.float64)
        idx = 0
        for j in range(n):
            if j == i:
                continue
            # Base slack: how much work j has relative to its processing time
            base_slack = wspt[j]
            # Blocking from i: proportional to duration overlap and i's duration
            blocking = dur_overlap[i, j] * min_durations[i] / (min_durations[j] + 1e-9)
            slacks[idx] = base_slack - blocking
            idx += 1
        
        # Minimum slack preserved by scheduling i
        min_slack_scores[i] = np.min(slacks) if len(slacks) > 0 else wspt[i]
    
    # Combine minimum slack maximization with WSPT as a tiebreaker
    n_min_slack = _normalize(min_slack_scores)
    
    w_min_slack = 0.55   # primary: maximize minimum slack (prevent starvation)
    w_wspt = 0.30        # secondary: urgency relative to processing time
    w_work = 0.15        # tertiary: total remaining work
    
    score = (w_min_slack * n_min_slack + 
             w_wspt * n_wspt + 
             w_work * nw)

    # Pick the operation with the highest score
    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
