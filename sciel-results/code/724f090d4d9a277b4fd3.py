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
    # Shift for numerical stability
    shifted = [v - v_min for v in values]
    log_sum = sum(math.exp(-beta * s) for s in shifted)
    if log_sum < 1e-300:
        return v_min
    return v_min - math.log(log_sum) / beta


def dispatch(ready_operations, time):
    """
    Pick the best ready operation to schedule next.
    
    Strategy: Multi-criteria heuristic combining:
    1. Dynamic Critical Ratio (CR) - urgency based on remaining work vs. time horizon
    2. Longest Remaining Processing Time (LRPT) - prioritize jobs with more work
    3. Shortest Processing Time (SPT) - minimize immediate machine occupation
    4. Job criticality - prioritize ops that are bottlenecks for their job
    5. Smooth machine contention - avoid heavily loaded machines
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
    job_ids = []
    op_indices = []
    
    for idx, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        mo = op.get("machine_options", {})
        job_id = op.get("job", idx)
        op_idx = op.get("op", idx)
        
        if mo:
            durations = list(mo.values())
            min_d = min(durations)
        else:
            min_d = 0
        
        remaining_work.append(rw)
        remaining_ops.append(ro)
        min_duration.append(min_d)
        job_ids.append(job_id)
        op_indices.append(op_idx)
    
    # Normalize features to [0, 1] range
    def normalize(vals):
        mn = min(vals)
        mx = max(vals)
        rng = mx - mn
        if rng < 1e-9:
            return [0.0] * len(vals)
        return [(v - mn) / rng for v in vals]
    
    norm_rw = normalize(remaining_work)
    norm_ro = normalize(remaining_ops)
    norm_min_d = normalize(min_duration)
    
    # Compute Dynamic Critical Ratio:
    # Estimate total remaining horizon as max(remaining_work) across all ready ops
    # CR = (horizon - remaining_work_i) / (remaining_work_i + 1)
    # Lower CR = more urgent (job has less slack relative to horizon)
    horizon = max(remaining_work) if remaining_work else 1.0
    horizon_safe = max(horizon, 1.0)
    
    cr_values = []
    for i in range(n):
        rw_safe = remaining_work[i] + 1.0
        # If remaining_work is close to horizon, CR is high (less urgent)
        # If remaining_work is small, CR is low (more urgent)
        cr = (horizon_safe - remaining_work[i]) / rw_safe
        cr_values.append(cr)
    
    norm_cr = normalize(cr_values)
    
    # Job criticality: for each job, if this is the last op, it's very critical
    # Also, jobs with fewer remaining ops are closer to completion
    job_remaining_ops = {}
    for i in range(n):
        j = job_ids[i]
        ro = remaining_ops[i]
        if j not in job_remaining_ops or ro < job_remaining_ops[j]:
            job_remaining_ops[j] = ro
    
    job_criticality = []
    for i in range(n):
        j = job_ids[i]
        ro = remaining_ops[i]
        min_ro_for_job = job_remaining_ops[j]
        # If this op has the minimum remaining ops for its job, it's critical
        # Also, lower remaining_ops = closer to done = more critical
        crit = 1.0 / (ro + 1.0) if ro > 0 else 2.0
        # Bonus if this is the bottleneck op for the job
        if ro == min_ro_for_job:
            crit *= 1.3
        job_criticality.append(crit)
    
    norm_crit = normalize(job_criticality)
    
    # Adaptive weights based on simulation time and number of ready ops
    # Early in simulation: prioritize long jobs (LRPT) to avoid stragglers
    # Late in simulation: prioritize short ops (SPT) and urgent jobs (CR)
    progress_ratio = time / max(horizon_safe + time, 1.0)
    
    # Base weights that shift with progress
    w_rw = 0.20 + 0.10 * (1 - progress_ratio)  # More LRPT early
    w_ro = 0.05
    w_spt = 0.30 + 0.15 * progress_ratio      # More SPT late
    w_cr = 0.25 + 0.10 * progress_ratio       # More CR late
    w_crit = 0.10
    
    # Adjust for number of ready ops
    if n <= 3:
        w_rw += 0.10
        w_spt -= 0.05
    elif n >= 8:
        w_spt += 0.05
        w_rw -= 0.05
    
    # Ensure weights sum to approximately 1
    total_w = w_rw + w_ro + w_spt + w_cr + w_crit
    w_rw /= total_w
    w_ro /= total_w
    w_spt /= total_w
    w_cr /= total_w
    w_crit /= total_w
    
    # Compute smooth machine contention using soft-min
    machine_counts = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m in mo:
            machine_counts[m] = machine_counts.get(m, 0) + 1
    
    max_contention = max(machine_counts.values()) if machine_counts else 1
    
    # Compute scores (higher is better for our selection)
    scores = []
    for i in range(n):
        s_rw = norm_rw[i]
        s_ro = norm_ro[i]
        # SPT: lower min_duration is better
        s_spt = 1.0 - norm_min_d[i]
        # CR: lower CR is more urgent, so higher score for lower CR
        s_cr = 1.0 - norm_cr[i]
        s_crit = norm_crit[i]
        
        # Smooth machine contention
        mo = ready_operations[i].get("machine_options", {})
        if mo:
            # Get contention levels for all machines this op can use
            contention_levels = [float(machine_counts.get(m, 1)) for m in mo]
            # Soft-min gives a smooth estimate of "best" contention
            soft_contention = _soft_min(contention_levels, beta=3.0)
            # Normalize: lower contention is better
            s_contention = 1.0 - (soft_contention - 1.0) / max(1.0, max_contention - 1.0)
        else:
            s_contention = 0.5
        
        score = (w_rw * s_rw + 
                 w_ro * s_ro + 
                 w_spt * s_spt + 
                 w_cr * s_cr +
                 w_crit * s_crit +
                 0.08 * s_contention)
        scores.append(score)
    
    # Select best with small tiebreaker for stability
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score + 1e-9:
            best_score = scores[i]
            best_idx = i
    
    return int(best_idx)

# EVOLVE-BLOCK-END
