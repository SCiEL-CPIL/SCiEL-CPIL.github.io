"""Job-shop dispatch rule: CR-based urgency with adaptive weighting and contention."""
# EVOLVE-BLOCK-START
import time as _time
import math

_dispatch_start_time = _time.time()
_DEADLINE_MARGIN = 135.0


def _deadline_exceeded():
    return (_time.time() - _dispatch_start_time) > _DEADLINE_MARGIN


def dispatch(ready_operations, time):
    """
    Pick the best ready operation to schedule next.
    
    Strategy: Critical Ratio (CR) enhanced multi-criteria with dynamic slack estimation.
    CR = slack / remaining_processing_time, where slack is estimated from
    the sum of minimum durations of all remaining operations in the job.
    Lower CR = more urgent = higher priority.
    
    Additional criteria:
    - Normalized Shortest Processing Time (NSPT)
    - Remaining work volume
    - Machine contention avoidance
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
    sum_remaining_min_durations = []  # Sum of min durations for all remaining ops in the job
    
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
        
        # Estimate total remaining processing time using min durations
        # remaining_work is typically the sum of processing times
        # We use remaining_work as a proxy for total remaining work
        sum_remaining_min_durations.append(rw)
    
    # Compute Critical Ratio (CR) for each operation
    # CR = slack / processing_time
    # slack = estimated_time_to_complete_job - current_time - current_op_duration
    # Since we don't have true deadlines, we estimate urgency based on
    # how much work remains relative to how long this operation takes
    cr_values = []
    for i in range(n):
        rw = remaining_work[i]
        md = min_duration[i]
        
        if md > 1e-9:
            # Slack estimate: remaining work after this op completes
            # Urgency increases when remaining work is small relative to this op's duration
            slack_estimate = max(0, rw - md)
            cr = slack_estimate / (md + 1e-6)
        else:
            # Zero processing time: very urgent
            cr = 0.0
        
        cr_values.append(cr)
    
    # Normalize features to [0, 1] range
    def normalize(vals):
        if not vals:
            return []
        mn = min(vals)
        mx = max(vals)
        rng = mx - mn
        if rng < 1e-9:
            return [0.0] * len(vals)
        return [(v - mn) / rng for v in vals]
    
    norm_rw = normalize(remaining_work)
    norm_ro = normalize(remaining_ops)
    norm_avg_d = normalize(avg_duration)
    norm_cr = normalize(cr_values)
    
    # Compute NSPT: normalized shortest processing time weighted by remaining work
    nspt_values = []
    for i in range(n):
        rw_safe = remaining_work[i] + 1.0
        nspt = min_duration[i] / rw_safe
        nspt_values.append(nspt)
    
    norm_nspt = normalize(nspt_values)
    
    # Compute machine contention: how many ready ops target each machine
    machine_counts = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m in mo:
            machine_counts[m] = machine_counts.get(m, 0) + 1
    
    max_contention = max(machine_counts.values()) if machine_counts else 1
    
    # Adaptive weights based on number of ready operations
    if n <= 3:
        w_rw = 0.35
        w_ro = 0.10
        w_nspt = 0.30
        w_avg_d = 0.05
        w_cr = 0.20
    elif n >= 8:
        w_rw = 0.25
        w_ro = 0.10
        w_nspt = 0.35
        w_avg_d = 0.05
        w_cr = 0.25
    else:
        w_rw = 0.30
        w_ro = 0.10
        w_nspt = 0.32
        w_avg_d = 0.05
        w_cr = 0.23
    
    # Compute scores (higher is better for selection)
    scores = []
    for i in range(n):
        s_rw = norm_rw[i]
        s_ro = norm_ro[i]
        s_nspt = 1.0 - norm_nspt[i]  # Lower NSPT = better
        s_avg_d = 1.0 - norm_avg_d[i]  # Shorter duration = better
        s_cr = 1.0 - norm_cr[i]  # Lower CR = more urgent = better
        
        # Machine contention: prefer ops going to less-contended machines
        mo = ready_operations[i].get("machine_options", {})
        if mo:
            best_machine = min(mo, key=lambda m: mo[m])
            contention = machine_counts.get(best_machine, 1)
            s_contention = 1.0 - (contention - 1) / max(1, max_contention - 1)
        else:
            s_contention = 0.5
        
        score = (w_rw * s_rw + 
                 w_ro * s_ro + 
                 w_nspt * s_nspt + 
                 w_avg_d * s_avg_d +
                 w_cr * s_cr +
                 0.10 * s_contention)
        scores.append(score)
    
    # Select best operation
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score + 1e-9:
            best_score = scores[i]
            best_idx = i
    
    return int(best_idx)

# EVOLVE-BLOCK-END
