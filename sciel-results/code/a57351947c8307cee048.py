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
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0 
        for op in ready_operations
    ], dtype=float)
    num_options = np.array([
        len(op["machine_options"]) if op["machine_options"] else 1 
        for op in ready_operations
    ], dtype=float)
    
    # Avoid division by zero
    num_options = np.maximum(num_options, 1)
    flexibility = 1.0 / num_options

    # Composite Score
    # Primary: Maximize remaining work (clear the biggest bottleneck)
    # Secondary: Minimize min_duration (free the machine quickly)
    # Tertiary: Maximize flexibility penalty (schedule constrained ops first)
    
    # Normalize features to prevent scale issues?
    # remaining_work and min_duration are in same units (time).
    # Weights:
    w_work = 1.0
    w_dur = 0.1
    w_flex = 0.05

    scores = (
        w_work * remaining_works 
        - w_dur * min_durations 
        + w_flex * flexibility
    )
    
    # Add small random noise for tie-breaking
    # This helps if multiple ops have identical scores, allowing the simulation
    # to potentially find a better global path by varying the choice.
    noise = np.random.uniform(0, 0.01, size=n)
    scores += noise

    best_idx = int(np.argmax(scores))
    
    # Select machine: Shortest duration to free the machine ASAP
    best_op = ready_operations[best_idx]
    machine_options = best_op.get("machine_options", {})
    if machine_options:
        best_machine = min(machine_options, key=machine_options.get)
        return (best_idx, best_machine)
    
    return best_idx
# EVOLVE-BLOCK-END
