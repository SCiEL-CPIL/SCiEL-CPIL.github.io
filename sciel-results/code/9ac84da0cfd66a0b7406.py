"""Open-shop scheduling dispatch rule.

Combines a "Most Critical Job" heuristic (max remaining work) with
machine selection (shortest processing time) to minimize makespan.
Uses NumPy for efficient vectorized scoring.

The scoring function approximates a load-balancing objective:
1. Prioritize operations from jobs with high remaining work (critical path).
2. Prefer operations that can be scheduled on a machine with short duration 
   (frees the machine quickly, reducing contention).
3. Include work density and flexibility as secondary factors.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.time()
_HARD_LIMIT = 140.0  # seconds


def _elapsed_ok():
    return (_time.time() - _START) < _HARD_LIMIT


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        machine_options = op.get("machine_options", {})
        if machine_options:
            # Return the machine with the shortest duration to free it ASAP
            best_machine = min(machine_options, key=machine_options.get)
            return (0, best_machine)
        return 0

    # Deadline check: if we are running out of time, use a simple O(n) fallback
    if not _elapsed_ok():
        max_work = -1
        best_idx = 0
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            if rw > max_work:
                max_work = rw
                best_idx = i
        op = ready_operations[best_idx]
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_machine = min(machine_options, key=machine_options.get)
            return (best_idx, best_machine)
        return best_idx

    # Extract features into NumPy arrays for vectorized computation
    remaining_works = np.array([op.get("remaining_work", 0) for op in ready_operations], dtype=float)
    remaining_ops = np.array([op.get("remaining_ops", 1) for op in ready_operations], dtype=float)
    
    # Calculate soft-min duration for each operation's machine options.
    # We use a sharp soft-min (beta=10) to closely approximate the true minimum
    # while maintaining smoothness. Also compute mean duration as a load proxy.
    min_durations = np.zeros(n, dtype=float)
    mean_durations = np.zeros(n, dtype=float)
    num_options = np.zeros(n, dtype=float)
    
    BETA = 10.0  # Sharp soft-min: closely approximates true min
    
    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if mo:
            vals = np.array(list(mo.values()), dtype=float)
            num_options[i] = len(vals)
            
            # Sharp soft-min: -1/beta * log(sum(exp(-beta * vals)))
            # = min(vals) - (1/beta) * log(sum(exp(beta * (min(vals) - vals))))
            min_val = np.min(vals)
            # exp(beta * (min_val - vals)) is always <= 1, so no overflow.
            exp_terms = np.exp(BETA * (min_val - vals))
            sum_exp = np.sum(exp_terms)
            if sum_exp > 0:
                log_sum_exp = np.log(sum_exp)
                min_durations[i] = min_val - log_sum_exp / BETA
            else:
                min_durations[i] = min_val
            
            mean_durations[i] = np.mean(vals)
        else:
            num_options[i] = 1.0
            min_durations[i] = 0.0
            mean_durations[i] = 0.0

    # Avoid division by zero for work density
    remaining_ops_safe = np.maximum(remaining_ops, 1.0)
    work_density = remaining_works / remaining_ops_safe

    # Flexibility: fewer options means more constrained, higher urgency
    num_options_safe = np.maximum(num_options, 1.0)
    flexibility = 1.0 / num_options_safe

    # Composite Score
    # We want to MAXIMIZE this score.
    # 
    # 1. remaining_works: Higher is better (critical path). Strongest signal.
    # 2. -min_durations: Lower duration is better (frees machine quickly).
    #    Sharp soft-min (beta=10) closely approximates true min.
    # 3. -mean_durations: Lower mean duration = less machine contention overall.
    # 4. work_density: Higher is better (urgency per remaining op).
    # 5. flexibility: Higher is better (constrained ops have less scheduling slack).
    
    # Weights tuned: emphasize critical path, then machine freedom, then urgency
    w_work = 1.0
    w_dur = 0.3
    w_mean = 0.05
    w_density = 0.1
    w_flex = 0.08

    scores = (
        w_work * remaining_works 
        - w_dur * min_durations 
        - w_mean * mean_durations
        + w_density * work_density
        + w_flex * flexibility
    )
    
    # Deterministic tie-breaking: prefer lower index
    best_idx = int(np.argmax(scores))
    
    # Select machine: Shortest duration to free the machine ASAP
    best_op = ready_operations[best_idx]
    machine_options = best_op.get("machine_options", {})
    if machine_options:
        best_machine = min(machine_options, key=machine_options.get)
        return (best_idx, best_machine)
    
    return best_idx
# EVOLVE-BLOCK-END
