"""Open-shop scheduling dispatch rule.

Combines a "Most Critical Job" heuristic (max remaining work) with
machine selection (shortest processing time) to minimize makespan.
Uses NumPy for efficient scoring and a small random tie-breaker to
escape deterministic local minima.
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
        # Return index and the machine with the shortest duration
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_machine = min(machine_options, key=machine_options.get)
            return (0, best_machine)
        return 0

    # Deadline check
    if not _elapsed_ok():
        # Fallback to simple rule: max remaining work
        max_work = max(op["remaining_work"] for op in ready_operations)
        for i, op in enumerate(ready_operations):
            if op["remaining_work"] == max_work:
                machine_options = op.get("machine_options", {})
                if machine_options:
                    best_machine = min(machine_options, key=machine_options.get)
                    return (i, best_machine)
                return i
        return 0

    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    
    # Soft-min of machine durations (smooth surrogate for hard min)
    # Using log-sum-exp with negative values: softmin(x) = -logsumexp(-x)
    min_durations = np.zeros(n, dtype=float)
    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if mo:
            vals = np.array(list(mo.values()), dtype=float)
            # Soft-min: smooth approximation of min
            beta = 5.0  # higher beta = closer to true min
            neg_vals = -vals
            max_neg = np.max(neg_vals)
            lse = max_neg + np.log(np.sum(np.exp(neg_vals - max_neg)))
            min_durations[i] = -lse
        else:
            min_durations[i] = 0.0
    
    num_options = np.array([
        len(op["machine_options"]) if op["machine_options"] else 1 
        for op in ready_operations
    ], dtype=float)
    
    # Avoid division by zero
    num_options_safe = np.maximum(num_options, 1)
    flexibility = 1.0 / num_options_safe
    
    # Work density: remaining work per remaining operation
    remaining_ops_safe = np.maximum(remaining_ops, 1)
    work_density = remaining_works / remaining_ops_safe
    
    # Composite Score
    # Primary: Maximize remaining work (clear the biggest bottleneck)
    # Secondary: Minimize soft-min duration (free the machine quickly)
    # Tertiary: Maximize work density (urgency per op)
    # Quaternary: Maximize flexibility penalty (schedule constrained ops first)
    
    # Weights tuned for balance
    w_work = 1.0
    w_dur = 0.15
    w_density = 0.3
    w_flex = 0.05
    
    scores = (
        w_work * remaining_works 
        - w_dur * min_durations 
        + w_density * work_density
        + w_flex * flexibility
    )
    
    # Deterministic tie-breaking: prefer lower index for consistency
    best_idx = int(np.argmax(scores))
    
    # Select machine: Shortest duration to free the machine ASAP
    best_op = ready_operations[best_idx]
    machine_options = best_op.get("machine_options", {})
    if machine_options:
        best_machine = min(machine_options, key=machine_options.get)
        return (best_idx, best_machine)
    
    return best_idx
# EVOLVE-BLOCK-END
