"""Job-shop scheduling dispatch rule."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.monotonic()
_DEADLINE = 140.0  # Safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    Combines the efficiency metric (remaining_work / min_duration) with
    machine-contention avoidance via flexibility ratio and a soft-min surrogate
    for the effective processing time.
    
    Key insights:
    - Efficiency ratio captures how "urgent" an operation is relative to job's total load
    - Flexibility ratio (min_dur / total_dur) identifies operations that can run on
      less-loaded machines, reducing contention
    - Soft-min via log-sum-exp provides a smooth surrogate for the minimum duration,
      capturing the "effective" processing time considering all machine options
    - SPT bonus favors short operations to free machines quickly
    - Remaining work prioritizes jobs with more total work
    """
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _time.monotonic() - _START_TIME > _DEADLINE:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        durations = list(machine_options.values()) if machine_options else [1.0]
        
        min_duration = min(durations) if durations else 1.0
        total_duration = sum(durations) if durations else 1.0
        num_machines = len(durations)
        
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        
        # Primary: efficiency metric - remaining work per unit of current processing time
        # High efficiency means the current step is short relative to total remaining work
        if min_duration > 0:
            efficiency = remaining_work / min_duration
        else:
            efficiency = remaining_work + 1000.0
        
        # Secondary: SPT bonus - pure shortest processing time preference
        if min_duration > 0:
            spt_score = 1.0 / min_duration
        else:
            spt_score = 1000.0
        
        # Machine-contention avoidance: flexibility ratio
        # min_dur / total_dur: lower ratio means more flexible (can pick shorter machine)
        # Higher ratio means more constrained (all machines have similar duration)
        if total_duration > 0:
            flexibility_ratio = min_duration / total_duration
        else:
            flexibility_ratio = 1.0
        
        # Inverse flexibility: operations with lower ratio (more flexible) get higher score
        # This prioritizes operations that can be executed on less loaded machines
        flexibility_score = 1.0 - flexibility_ratio  # Higher = more flexible
        
        # Number of machine options: more options = less contention
        num_options_score = float(num_machines)
        
        # Tertiary: remaining work - prioritize jobs with more total work
        work_score = remaining_work
        
        # Quaternary: fewer remaining ops means job is closer to completion
        ops_score = -remaining_ops
        
        # Soft-min surrogate: smooth approximation of the minimum duration using
        # log-sum-exp. This captures the "effective" processing time considering
        # all machine options, not just the absolute minimum.
        # soft_min = -log(sum(exp(-d / tau)) * tau) where tau is temperature
        # For small tau, this approaches the true min; for larger tau, it's smoother.
        # We use a moderate temperature to balance smoothness and fidelity.
        if durations:
            tau = 2.0  # Temperature parameter for soft-min
            # soft_min = -tau * log(sum(exp(-d/tau)))
            # To avoid overflow, use the log-sum-exp trick
            neg_d_over_tau = np.array([-d / tau for d in durations])
            max_neg = np.max(neg_d_over_tau)
            log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_d_over_tau - max_neg)))
            soft_min = -tau * log_sum_exp
            
            # The difference between soft_min and hard min captures the "spread"
            # of durations. If all machines have similar times, soft_min ≈ min_duration.
            # If there's a wide spread, soft_min > min_duration, indicating that
            # the operation is less flexible in practice.
            spread_penalty = max(0.0, soft_min - min_duration)
        else:
            spread_penalty = 0.0
        
        # Composite score:
        # Efficiency is primary (captures SPT + remaining work interaction)
        # SPT provides additional weight to very short operations
        # Flexibility avoids machine contention
        # Spread penalty from soft-min captures practical flexibility
        # Remaining work and ops provide secondary effects
        scores[i] = (
            1.0 * efficiency +          # Primary: efficiency ratio
            2.0 * spt_score +           # Secondary: pure SPT
            0.5 * flexibility_score +   # Machine-contention avoidance
            0.1 * num_options_score +   # More machine options = less contention
            0.1 * work_score +          # Tertiary: remaining work
            0.5 * ops_score -           # Quaternary: fewer remaining ops
            0.3 * spread_penalty        # Soft-min spread penalty
        )
    
    # Select the operation with the highest score
    best_idx = int(np.argmax(scores))
    
    return int(best_idx)
# EVOLVE-BLOCK-END
