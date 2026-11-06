"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
import math

_START_TIME = None

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Safety guard: if we are approaching the 120s limit, fall back to a fast heuristic
    # to ensure we don't timeout. The deadline is 120s, we check against 115s.
    if _time.time() - _START_TIME > 115:
        # Fast fallback: pick the operation with the shortest duration on its preferred machine
        # to keep the system moving.
        min_dur = float('inf')
        best_idx = 0
        for i, op in enumerate(ready_operations):
            # Find min duration for this op
            min_op_dur = min(op["machine_options"].values())
            if min_op_dur < min_dur:
                min_dur = min_op_dur
                best_idx = i
        return int(best_idx)

    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0
    if n_ops == 1:
        return 0

    # 1. Estimate machine load based on the current ready queue.
    # A simple proxy for "machine congestion" is the sum of durations of all ready ops
    # that can run on that machine.
    machine_load = {}
    for op in ready_operations:
        for m, d in op["machine_options"].items():
            if m not in machine_load:
                machine_load[m] = 0
            machine_load[m] += d

    # 2. Use a soft-max (log-sum-exp) to smoothly aggregate machine scores,
    # which avoids the discontinuity of hard max and better balances urgency vs. congestion.
    # Also add a critical-path urgency boost: jobs with fewer remaining operations
    # are closer to completion and should be prioritized to reduce makespan.
    
    best_score = -float('inf')
    best_idx = 0
    
    # Temperature for soft-max: lower = more like hard max, higher = more uniform
    # We use a moderate temperature to allow some smoothing
    temp = 0.5
    
    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        remaining_ops = op.get("remaining_ops", 1)
        
        if rw == 0:
            # Should not happen for ready ops, but safety
            score = -float('inf')
        else:
            # Compute raw scores for each machine option
            scores = []
            for m, d in op["machine_options"].items():
                load = machine_load.get(m, 0)
                # Use a softened denominator: add a small constant to avoid zero-division
                # and to make the score less extreme when load is very low
                denom = load + d + 1.0
                s = rw / denom
                scores.append(s)
            
            if not scores:
                score = -float('inf')
            else:
                # Soft-max via log-sum-exp for numerical stability
                max_s = max(scores)
                if max_s <= 0:
                    # All scores are non-positive; use the max directly
                    smax = max_s
                else:
                    # log-sum-exp: log(sum(exp(s_i))) ≈ max_s + log(sum(exp(s_i - max_s)))
                    exp_sum = 0.0
                    for s in scores:
                        exp_sum += math.exp(s - max_s)
                    smax = max_s + math.log(exp_sum)
                
                # Normalize by number of options to get an average-like soft-max
                soft_max_score = smax - math.log(len(scores))
                
                # Critical path urgency: prioritize jobs with fewer remaining operations
                # This encourages completing jobs that are close to done, reducing makespan
                # Use a smoother urgency function: inverse of remaining_ops with a floor
                urgency_boost = 1.0 / (1.0 + remaining_ops * 0.2)
                
                # Combine: weighted sum of soft-max machine score and urgency
                # Increase weight on machine score to better balance load
                score = soft_max_score * 0.8 + urgency_boost * 0.2 * (rw / (rw + 1))
        
        if score > best_score:
            best_score = score
            best_idx = i
            
    return int(best_idx)
# EVOLVE-BLOCK-END
