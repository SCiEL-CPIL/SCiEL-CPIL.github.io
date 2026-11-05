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
    
    # Calculate min and mean durations
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else np.inf
        for op in ready_operations
    ])
    
    # Use mean duration for efficiency to account for flexibility/robustness
    # Operations that are fast on average are preferred over those that are fast only on one machine
    mean_durations = np.array([
        np.mean(list(op["machine_options"].values())) if op["machine_options"] else np.inf
        for op in ready_operations
    ])

    # Compute critical path-like estimate
    avg_work_per_op = np.where(remaining_ops > 0, remaining_works / np.maximum(remaining_ops, 1), remaining_works)
    
    # Normalization
    max_work = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    max_avg = np.max(avg_work_per_op) if np.max(avg_work_per_op) > 0 else 1.0
    
    # For efficiency, we use 1/mean_duration.
    # We need to handle inf (no machine options)
    valid_mean_durations = mean_durations[mean_durations < np.inf]
    if len(valid_mean_durations) == 0:
        # All invalid? Fallback to index 0
        return 0
        
    max_inv_mean_dur = np.max(1.0 / valid_mean_durations) if np.any(valid_mean_durations > 0) else 1.0
    
    normalized_urgency = remaining_works / (max_work + 1e-9)
    normalized_avg = avg_work_per_op / (max_avg + 1e-9)
    
    # Efficiency based on mean duration
    # Replace inf with 0 efficiency
    inv_mean_durations = np.where(mean_durations < np.inf, 1.0 / mean_durations, 0.0)
    normalized_efficiency = inv_mean_durations / (max_inv_mean_dur + 1e-9)
    
    # Weights: 
    # Shift weight towards Urgency (Total Work) and away from pure Speed (Min Duration)
    # Using Mean Duration for efficiency makes it a "Robustness" metric
    w_urgency = 1.2
    w_avg = 0.5
    w_efficiency = 0.3
    
    scores = (w_urgency * normalized_urgency + 
              w_avg * normalized_avg + 
              w_efficiency * normalized_efficiency)

    best_idx = int(np.argmax(scores))

    # Choose the fastest machine for the selected operation (SPT)
    machine_options = ready_operations[best_idx]["machine_options"]
    if machine_options:
        best_machine = min(machine_options, key=machine_options.get)
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
