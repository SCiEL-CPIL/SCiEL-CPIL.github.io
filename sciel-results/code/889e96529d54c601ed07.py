"""Job-shop dispatch rule: refined multi-criteria with smooth contention and dynamic CR."""
# EVOLVE-BLOCK-START
import time as _time
import math

_dispatch_start_time = _time.time()
_DEADLINE_MARGIN = 135.0  # safety margin before 150s kill


def _deadline_exceeded():
    return (_time.time() - _dispatch_start_time) > _DEADLINE_MARGIN


def _soft_min(values, beta=5.0):
    """Soft-min via log-sum-exp for smoothness."""
    if not values:
        return 0.0
    v_min = min(values)
    shifted = [v - v_min for v in values]
    log_sum = sum(math.exp(-beta * s) for s in shifted)
    if log_sum < 1e-300:
        return v_min
    return v_min - math.log(log_sum) / beta


def dispatch(ready_operations, time):
    """
    Pick the best ready operation to schedule next.
    
    Strategy: Multi-criteria heuristic combining:
    1. Critical Ratio (CR) - urgency based on slack vs remaining processing time
    2. Longest Remaining Processing Time (LRPT) - prioritize jobs with more work
    3. Shortest Processing Time on assigned machine (SPT) - minimize immediate delay
    4. Most operations remaining - prioritize jobs with longer chains
    5. Machine contention - avoid scheduling to heavily loaded machines
    """
    if not ready_operations:
        return 0
    
    if _deadline_exceeded():
        return 0
    
    n = len(ready_operations)
    
    # Precompute features for each ready operation
    remaining_work = []
    remaining_ops = []
    min_duration = []
    max_duration = []
    avg_duration = []
    
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        mo = op.get("machine_options", {})
        
        if mo:
            durations = list(mo.values())
            min_d = min(durations)
            max_d = max(durations)
            avg_d = sum(durations) / len(durations)
        else:
            min_d = 0
            max_d = 0
            avg_d = 0
        
        remaining_work.append(rw)
        remaining_ops.append(ro)
        min_duration.append(min_d)
        max_duration.append(max_d)
        avg_duration.append(avg_d)
    
    # Normalize features to [0, 1] range for fair weighting
    def normalize(vals):
        mn = min(vals)
        mx = max(vals)
        rng = mx - mn
        if rng < 1e-9:
            return [0.0] * len(vals)
        return [(v - mn) / rng for v in vals]
    
    norm_rw = normalize(remaining_work)
    norm_ro = normalize(remaining_ops)
    norm_avg_d = normalize(avg_duration)
    norm_min_d = normalize(min_duration)
    
    # Compute NSPT: normalized shortest processing time weighted by remaining work
    # Lower NSPT = shorter duration relative to remaining work = better
    nspt_values = []
    for i in range(n):
        # Use min_duration normalized by remaining_work to get NSPT
        # Add epsilon to avoid division by zero
        rw_safe = remaining_work[i] + 1.0
        nspt = min_duration[i] / rw_safe
        nspt_values.append(nspt)
    
    norm_nspt = normalize(nspt_values)
    # For NSPT, lower is better, so we negate for scoring
    
    # Compute a meaningful urgency metric: ratio of remaining ops to remaining work
    # Jobs with many ops but little remaining work are "tight" and urgent
    # Also consider: ops with few remaining ops should be completed soon to free the job
    urgency_values = []
    for i in range(n):
        rw_safe = remaining_work[i] + 1.0
        ro_safe = remaining_ops[i] + 1.0
        # Higher urgency = more ops relative to work (tight schedule)
        # Also: fewer remaining ops = closer to completion = should finish
        urgency = (ro_safe / rw_safe) * 10.0 + (1.0 / ro_safe) * 5.0
        urgency_values.append(urgency)
    
    norm_urgency = normalize(urgency_values)
    
    # Adaptive weights based on situation
    # Base weights
    w_rw = 0.22
    w_ro = 0.08
    w_nspt = 0.42
    w_avg_d = 0.08
    w_urgency = 0.20
    
    # If few ready ops, be more aggressive with remaining work and urgency
    if n <= 3:
        w_rw = 0.32
        w_ro = 0.08
        w_nspt = 0.30
        w_avg_d = 0.08
        w_urgency = 0.22
    elif n >= 8:
        w_nspt = 0.36
        w_rw = 0.26
        w_ro = 0.08
        w_avg_d = 0.08
        w_urgency = 0.22
    
    # Compute machine contention using soft-min for smoothness
    # For each machine, compute how "loaded" it is based on ready ops targeting it
    machine_counts = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m in mo:
            machine_counts[m] = machine_counts.get(m, 0) + 1
    
    # Compute scores (higher is better for our selection)
    scores = []
    for i in range(n):
        # More remaining work = higher priority
        s_rw = norm_rw[i]
        # More remaining ops = higher priority  
        s_ro = norm_ro[i]
        # NSPT: lower normalized value = shorter duration relative to remaining work = better
        s_nspt = 1.0 - norm_nspt[i]
        s_avg_d = 1.0 - norm_avg_d[i]
        # Urgency: higher normalized value = more urgent = better
        s_urgency = norm_urgency[i]
        
        # Machine contention: use soft-min over machine durations to get smooth best-case
        mo = ready_operations[i].get("machine_options", {})
        if mo:
            durations = list(mo.values())
            # Soft-min gives a smooth approximation of the best (shortest) duration
            sm_dur = _soft_min(durations, beta=3.0)
            # Also compute contention on the best machine
            best_machine = min(mo, key=lambda m: mo[m])
            contention = machine_counts.get(best_machine, 1)
            max_contention = max(machine_counts.values()) if machine_counts else 1
            s_contention = 1.0 - (contention - 1) / max(1, max_contention - 1)
            # Combine: prefer shorter effective duration and less contention
            s_machine = 0.6 * (1.0 - norm_min_d[i]) + 0.4 * s_contention
        else:
            s_machine = 0.5
        
        score = (w_rw * s_rw + 
                 w_ro * s_ro + 
                 w_nspt * s_nspt + 
                 w_avg_d * s_avg_d +
                 w_urgency * s_urgency +
                 0.05 * s_machine)
        scores.append(score)
    
    # Add a small tiebreaker: prefer lower op index for stability
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score + 1e-9:
            best_score = scores[i]
            best_idx = i
    
    return int(best_idx)

# EVOLVE-BLOCK-END
