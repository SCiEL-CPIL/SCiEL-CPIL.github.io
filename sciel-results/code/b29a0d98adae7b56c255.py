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
    
    # Handle inf durations (should not happen if machine_options is non-empty)
    min_durations = np.where(np.isinf(min_durations), 1.0, min_durations)
    
    # Efficiency: Inverse of min duration (SPT)
    inv_min_durations = 1.0 / np.maximum(min_durations, 1e-9)
    
    # Normalize
    max_work = np.max(remaining_works)
    if max_work == 0:
        max_work = 1.0
    max_inv_dur = np.max(inv_min_durations)
    if max_inv_dur == 0:
        max_inv_dur = 1.0
        
    norm_work = remaining_works / max_work
    norm_eff = inv_min_durations / max_inv_dur
    
    # Weights: Strong MWS, Moderate SPT
    w_work = 5.0
    w_eff = 1.0
    
    scores = w_work * norm_work + w_eff * norm_eff
    
    best_idx = int(np.argmax(scores))
    
    # Choose the best machine for the selected operation (SPT)
    machine_options = ready_operations[best_idx]["machine_options"]
    if machine_options:
        best_machine = min(machine_options, key=lambda m: (machine_options[m], m))
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
