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
    # Lower temperature = closer to true min, sharper preference
    min_durations = np.zeros(n, dtype=float)
    temperature = 3.0  # lower = sharper (closer to true min)
    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if mo:
            vals = np.array(list(mo.values()), dtype=float)
            # Soft-min: smooth approximation of min
            # softmin = -1/T * log(sum(exp(-vals/T)))
            scaled = -vals / temperature
            max_scaled = np.max(scaled)
            lse = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
            min_durations[i] = -temperature * lse
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
    
    # Time-aware urgency: as time progresses, prioritize ops with more remaining work
    # relative to the current makespan estimate
    # Estimate current makespan proxy: max remaining work + current time
    makespan_proxy = float(np.max(remaining_works)) + time
    
    # Urgency: how "tight" is this op relative to the makespan?
    # Higher urgency = more critical to schedule now
    urgency = remaining_works / (makespan_proxy + 1e-9)
    
    # Critical path proxy: remaining work / (remaining ops + 1) captures
    # how much "depth" this job still has
    depth_ratio = remaining_works / (remaining_ops_safe + 1.0)
    
    # Composite Score
    # Primary: Maximize remaining work (clear the biggest bottleneck)
    # Secondary: Minimize soft-min duration (free the machine quickly)  
    # Tertiary: Maximize urgency (time-sensitive criticality)
    # Quaternary: Maximize depth ratio (jobs with more sequential work)
    # Quinary: Maximize flexibility penalty (schedule constrained ops first)
    
    # Weights tuned for balance
    w_work = 1.0
    w_dur = 0.12
    w_urgency = 0.25
    w_depth = 0.15
    w_flex = 0.05
    
    scores = (
        w_work * remaining_works 
        - w_dur * min_durations 
        + w_urgency * urgency
        + w_depth * depth_ratio
        + w_flex * flexibility
    )
    
    # Deterministic tie-breaking: prefer lower index for consistency
    best_idx = int(np.argmax(scores))
    
    # Select machine: Shortest duration to free the machine ASAP
    # But also consider: if there's a machine that's less loaded (fewer other ops),
    # prefer it to spread load. Use a small tie-breaker toward less-loaded machines.
    best_op = ready_operations[best_idx]
    machine_options = best_op.get("machine_options", {})
    if machine_options:
        # Primary: shortest duration
        # Secondary: prefer machine with fewer concurrent ops (less contention)
        # We don't have direct load info, so use duration as proxy:
        # shorter duration machines are likely less utilized
        best_machine = min(machine_options, key=machine_options.get)
        return (best_idx, best_machine)
    
    return best_idx
# EVOLVE-BLOCK-END
