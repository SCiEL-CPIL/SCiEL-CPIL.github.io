"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np
import math


# Global state to cache parameters and track time
_start_time = _time.time()
_DEADLINE = 150.0  # Safety margin before 150s kill


def _soft_min(durations, temperature=0.5):
    """
    Smooth surrogate for min using log-sum-exp.
    As temperature -> 0, this approaches min.
    """
    if not durations:
        return 0.0
    arr = np.array(durations, dtype=np.float64)
    if arr.size == 1:
        return float(arr[0])
    
    # Soft min: -T * log(sum(exp(-x/T)))
    # This gives a smooth approximation to min(x)
    scaled = -arr / temperature
    max_scaled = np.max(scaled)
    if max_scaled > 500:  # Avoid overflow
        return float(np.min(arr))
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return float(-temperature * log_sum)


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard: if we're close to the limit, use a very fast fallback
    elapsed = _time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Fallback: pick the operation with minimum duration
        best_index = 0
        min_dur = float('inf')
        for i, op in enumerate(ready_operations):
            machine_options = op.get("machine_options", {})
            if not machine_options:
                continue
            md = min(machine_options.values())
            if md < min_dur:
                min_dur = md
                best_index = i
        return int(best_index)

    # Refined bottleneck mitigation via Workload, Constraint Tightness, and Smooth Duration
    # Key improvements:
    # 1. Use soft-min (log-sum-exp) for effective processing time - captures uncertainty
    #    in machine choice smoothly rather than hard min
    # 2. Incorporate remaining operations count as a critical-path proxy
    # 3. Better weighting: workload * tightness / effective_duration
    
    best_score = -float('inf')
    best_index = 0
    
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue
        
        # Constraint tightness: fewer machine options = more constrained
        num_options = len(machine_options)
        durations = list(machine_options.values())
        min_duration = min(durations)
        
        # Soft-min: smooth surrogate that accounts for all machine options
        # Lower temperature = closer to hard min; higher = more "average-like"
        eff_duration = _soft_min(durations, temperature=0.3)
        
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 1)
        
        # Critical path estimate: remaining work spread over remaining ops
        # This approximates the "urgency" of this job's critical path
        if remaining_ops > 0:
            critical_path_est = remaining_work / remaining_ops
        else:
            critical_path_est = 0.0
        
        # Score components:
        # 1. Workload urgency: high remaining work / few options = critical
        # 2. Speed: shorter effective duration is better
        # 3. Critical path: jobs with longer critical paths are more bottleneck-prone
        
        # Weighted combination:
        # - Primary: workload weighted by tightness (constrained big jobs first)
        # - Secondary: inverse effective duration (faster is better)
        # - Tertiary: critical path estimate (longer = more urgent)
        
        if remaining_work == 0:
            # No remaining work: prioritize tight and fast
            score = (1.0 / num_options) * 2.0 + (1.0 / max(eff_duration, 1e-9))
        else:
            # Main scoring: workload * tightness / effective_duration
            # This rewards high-work, constrained jobs that can be done quickly
            base_score = (remaining_work / num_options) / max(eff_duration, 1e-9)
            
            # Add critical path urgency as a multiplier-like bonus
            cp_bonus = critical_path_est * 0.1
            
            # Small speed bonus to break ties
            speed_bonus = 1.0 / max(eff_duration, 1e-9) * 0.05
            
            score = base_score + cp_bonus + speed_bonus
            
        if score > best_score:
            best_score = score
            best_index = i
            
    return int(best_index)
# EVOLVE-BLOCK-END
