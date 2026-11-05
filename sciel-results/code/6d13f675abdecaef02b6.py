"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_DEADLINE_SECONDS = 145.0  # Safety margin before 150s kill

# Track machine loads to estimate contention
_machine_loads = {}


def _smooth_min(durations, temperature=1.0):
    """
    Soft-min via log-sum-exp: approximates min(durations) smoothly.
    For a list of positive durations, this gives a smooth surrogate for min.
    """
    arr = np.asarray(durations, dtype=np.float64)
    if arr.size == 0:
        return 1e9
    m = np.min(arr)
    # Use soft-min: m - temperature * log(sum(exp(-(d - m)/temperature)))
    # This is smooth and approximates min as temperature -> 0
    shifted = (arr - m) / temperature
    # For numerical stability, use log-sum-exp
    max_shifted = np.max(shifted)
    lse = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return m - temperature * lse


def _compute_op_score(op, time):
    """
    Compute a priority score for a ready operation.
    Higher score means higher priority (start first).
    
    Strategy: Balance job-centric remaining work with machine contention.
    - Prioritize jobs with more remaining work (LRPT-like)
    - Favor operations on machines with lower current load (contention-aware)
    - Use smooth min for duration to avoid nonsmooth penalties
    """
    work = op.get("remaining_work", 0)
    machine_options = op.get("machine_options", {})
    
    if not machine_options:
        return 0.0
    
    durations = list(machine_options.values())
    machines = list(machine_options.keys())
    
    # Smooth min duration (soft surrogate for min)
    soft_min_dur = _smooth_min(durations, temperature=0.5)
    
    remaining_ops = op.get("remaining_ops", 0)
    
    # Estimate machine contention: average load on candidate machines
    # Higher load means more congestion -> we want to AVOID high-load machines
    # So we prefer ops where the best (shortest) machine has lower load
    if machines:
        avg_load = sum(_machine_loads.get(m, 0.0) for m in machines) / len(machines)
    else:
        avg_load = 0.0
    
    # Score components:
    # 1. Remaining work (primary driver for makespan)
    score = 1.0 * work
    
    # 2. Prefer shorter effective duration (smooth min)
    score += 0.8 / (soft_min_dur + 1e-9)
    
    # 3. Prefer lower machine contention (avoid congested machines)
    #    Negative penalty for high average load
    score -= 0.6 * avg_load / (1.0 + avg_load)
    
    # 4. Prefer jobs closer to completion (fewer remaining ops)
    score += 0.3 / (remaining_ops + 1e-9)
    
    return score


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Returns: index of the ready operation to start.
    """
    if not ready_operations:
        return 0
    
    if len(ready_operations) == 1:
        return 0
    
    # Deadline guard
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        return 0
    
    # Compute scores for all ready operations
    scores = [_compute_op_score(op, time) for op in ready_operations]
    
    # Select the operation with the highest score
    best_idx = int(np.argmax(scores))
    
    # Update machine load tracking for the selected operation
    # We estimate that starting this op will add its min duration to the best machine
    selected_op = ready_operations[best_idx]
    machine_options = selected_op.get("machine_options", {})
    if machine_options:
        # Find the machine with the shortest duration (likely the one we'll use)
        best_machine = min(machine_options, key=machine_options.get)
        dur = machine_options[best_machine]
        _machine_loads[best_machine] = _machine_loads.get(best_machine, 0.0) + dur
    
    return best_idx
# EVOLVE-BLOCK-END
