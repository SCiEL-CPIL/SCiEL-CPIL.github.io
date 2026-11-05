"""Job-Shop Scheduling dispatch rule: Critical Ratio (CR) heuristic with SPT tie-break."""
# EVOLVE-BLOCK-START
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE_MARGIN = 140.0  # safety margin before 150s kill


def _deadline_exceeded():
    return (_time.time() - _dispatch_start_time) > _DEADLINE_MARGIN


def dispatch(ready_operations, time):
    """
    Pick the best ready operation to schedule next.
    
    Strategy: Critical Ratio (CR) heuristic.
    CR = slack / remaining_processing_time
    
    where:
      - slack = estimated time left until the job must finish
      - remaining_processing_time = sum of min durations for all remaining ops in the job
    
    Lower CR means higher urgency (less slack relative to work remaining).
    We prioritize operations with the lowest CR to reduce makespan by
    focusing on jobs that are closest to their deadline.
    
    Tie-breaking: among operations with similar CR, prefer shorter
    processing time (SPT) to minimize idle time.
    """
    if not ready_operations:
        return 0
    
    if _deadline_exceeded():
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Precompute features for each ready operation
    cr_values = []          # Critical Ratio: slack / remaining_work
    min_duration = []       # shortest duration for this operation
    remaining_work = []     # total remaining work for the job
    remaining_ops = []      # number of remaining operations
    
    # Estimate the makespan target: use the maximum remaining work across jobs
    # as a proxy for the deadline. This gives a dynamic slack estimate.
    max_remaining_work = 0
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        if rw > max_remaining_work:
            max_remaining_work = rw
    
    # If no work remaining, just pick the first
    if max_remaining_work <= 0:
        return 0
    
    for op in ready_operations:
        mo = op.get("machine_options", {})
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        
        if mo:
            durations = list(mo.values())
            min_d = min(durations)
        else:
            min_d = float('inf')
        
        # CR = slack / remaining_work
        # slack = (estimated makespan target - current time)
        # We use max_remaining_work as a proxy for the makespan target,
        # so slack = max_remaining_work - time (capped at 0)
        slack = max(0.0, max_remaining_work - time)
        
        # Avoid division by zero
        if rw > 1e-9:
            cr = slack / rw
        else:
            # No remaining work: very urgent (CR = 0)
            cr = 0.0
        
        cr_values.append(cr)
        min_duration.append(min_d)
        remaining_work.append(rw)
        remaining_ops.append(ro)
    
    # Normalize CR values to [0, 1] for consistent weighting
    finite_cr = [c for c in cr_values if c != float('inf')]
    if finite_cr:
        cr_min = min(finite_cr)
        cr_max = max(finite_cr)
        cr_range = cr_max - cr_min
        if cr_range < 1e-9:
            norm_cr = [0.5] * n
        else:
            norm_cr = [(c - cr_min) / cr_range if c != float('inf') else 1.0 for c in cr_values]
    else:
        norm_cr = [0.5] * n
    
    # Normalize min_duration to [0, 1]
    finite_dur = [d for d in min_duration if d != float('inf')]
    if finite_dur:
        dur_min = min(finite_dur)
        dur_max = max(finite_dur)
        dur_range = dur_max - dur_min
        if dur_range < 1e-9:
            norm_dur = [0.5] * n
        else:
            norm_dur = [(d - dur_min) / dur_range if d != float('inf') else 1.0 for d in min_duration]
    else:
        norm_dur = [0.5] * n
    
    # Composite score: lower is better
    # Primary: CR (lower CR = more urgent = better)
    # Secondary: SPT (shorter duration = better)
    w_cr = 0.70
    w_spt = 0.30
    
    best_idx = 0
    best_score = float('inf')
    
    for i in range(n):
        # Lower CR is better
        s_cr = norm_cr[i]
        
        # Shorter duration is better
        s_spt = norm_dur[i]
        
        # Composite: lower is better
        score = w_cr * s_cr + w_spt * s_spt
        
        # Penalty for no machine options
        if not ready_operations[i].get("machine_options", {}):
            score += 0.1
        
        if score < best_score - 1e-12:
            best_score = score
            best_idx = i
    
    return int(best_idx)

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-END
