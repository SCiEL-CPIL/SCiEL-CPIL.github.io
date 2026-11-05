"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    elapsed = _time.time() - _dispatch_start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: pick operation with most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        machine_options = op["machine_options"]
        if machine_options:
            best_machine = min(machine_options, key=machine_options.get)
            return (0, best_machine)
        return 0

    # Extract key features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else np.inf
        for op in ready_operations
    ], dtype=float)
    
    # Handle inf durations
    min_durations = np.where(np.isinf(min_durations), 1.0, min_durations)
    
    # Build machine contention map: for each machine, count how many ready ops can use it
    machine_contention = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_contention[m] = machine_contention.get(m, 0) + 1
    
    # For each operation, compute a congestion-adjusted score
    # Score = (remaining_work / min_duration) / (1 + alpha * avg_contention_of_eligible_machines)
    # Higher alpha = more penalty for congested machines
    alpha = 0.5
    
    scores = np.zeros(n, dtype=float)
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if not mo:
            scores[i] = remaining_works[i] / (min_durations[i] + 1e-9)
            continue
        
        # Average contention across this op's eligible machines
        avg_contention = np.mean([machine_contention.get(m, 1) for m in mo.keys()])
        
        # Base urgency: remaining work relative to fastest option
        base_score = remaining_works[i] / (min_durations[i] + 1e-9)
        
        # Congestion penalty: divide by (1 + alpha * avg_contention)
        # This favors ops that can run on less-contended machines
        congestion_factor = 1.0 + alpha * avg_contention
        scores[i] = base_score / congestion_factor
    
    best_idx = int(np.argmax(scores))
    
    # Choose the best machine for the selected operation
    # Use a composite criterion: prefer shorter duration, then less contention
    machine_options = ready_operations[best_idx]["machine_options"]
    if machine_options:
        # Score each machine: primary = duration, secondary = -contention (prefer less loaded)
        def machine_score(m):
            dur = machine_options[m]
            cont = machine_contention.get(m, 1)
            # Minimize duration, break ties by lower contention
            return (dur, cont)
        best_machine = min(machine_options.keys(), key=machine_score)
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
