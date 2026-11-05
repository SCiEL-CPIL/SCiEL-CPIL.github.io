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

    # Extract key features for all ready operations
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else np.inf
        for op in ready_operations
    ])
    
    # Compute critical path-like estimate: remaining work divided by number of remaining ops
    # This gives an average work per remaining operation
    avg_work_per_op = np.where(remaining_ops > 0, remaining_works / np.maximum(remaining_ops, 1), remaining_works)
    
    # Compute the effective duration on the best machine (shortest processing time)
    # This represents how quickly this operation can be done
    
    max_work = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    max_avg = np.max(avg_work_per_op) if np.max(avg_work_per_op) > 0 else 1.0
    max_inv_dur = np.max(1.0 / min_durations) if np.any(min_durations > 0) else 1.0
    
    # Normalize components to [0, 1]
    normalized_urgency = remaining_works / (max_work + 1e-9)
    normalized_avg = avg_work_per_op / (max_avg + 1e-9)
    normalized_efficiency = (1.0 / min_durations) / (max_inv_dur + 1e-9)
    
    # Weighted score combining:
    # - Urgency: total remaining work (critical path length proxy)
    # - Depth-adjusted urgency: average work per remaining op (prioritizes jobs with high-intensity remaining work)
    # - Efficiency: how fast the operation can be completed on its best machine
    w_urgency = 1.0
    w_avg = 0.5
    w_efficiency = 0.3
    scores = w_urgency * normalized_urgency + w_avg * normalized_avg + w_efficiency * normalized_efficiency

    best_idx = int(np.argmax(scores))

    # Choose the fastest machine for the selected operation
    machine_options = ready_operations[best_idx]["machine_options"]
    if machine_options:
        best_machine = min(machine_options, key=machine_options.get)
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
