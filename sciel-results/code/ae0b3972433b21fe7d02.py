"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global deadline guard
_START_TIME = _time.time()
_DEADLINE = 140.0  # 10s safety margin before 150s

def dispatch(ready_operations, time):
    # Check deadline
    if _time.time() > _DEADLINE:
        # Fallback: Return first index to prevent crash/timeout score 0
        return 0

    if not ready_operations:
        return 0

    # Strategy: Normalized Urgency + Efficiency
    # 1. Normalize remaining_work to [0, 1] to prevent scale dominance.
    # 2. Normalize efficiency (1/min_duration) to [0, 1].
    # 3. Weighted sum: 1.0 * Urgency + 0.4 * Efficiency.
    # 4. Select operation with max score.
    # 5. Select machine with min duration for that operation.
    
    n_ops = len(ready_operations)
    
    # Pre-compute min_duration and remaining_work for all ops
    min_durations = []
    remaining_works = []
    best_machines = []
    
    for op in ready_operations:
        rem_work = op["remaining_work"]
        machine_opts = op["machine_options"]
        
        min_dur = float('inf')
        best_m = None
        for m, d in machine_opts.items():
            if d < min_dur:
                min_dur = d
                best_m = m
        
        min_durations.append(min_dur)
        remaining_works.append(rem_work)
        best_machines.append(best_m)
    
    # Handle edge case where no machine options? (Should not happen)
    if not min_durations:
        return 0

    max_rem_work = max(remaining_works) if remaining_works else 1
    if max_rem_work == 0:
        max_rem_work = 1

    # Calculate inverse durations for efficiency
    inv_durations = [1.0 / d if d > 0 else 0.0 for d in min_durations]
    max_inv_dur = max(inv_durations) if inv_durations else 1
    if max_inv_dur == 0:
        max_inv_dur = 1

    best_score = -1
    best_idx = 0
    best_machine = None

    for i in range(n_ops):
        norm_urgency = remaining_works[i] / max_rem_work
        norm_efficiency = inv_durations[i] / max_inv_dur
        
        # Weighted combination
        score = 1.0 * norm_urgency + 0.4 * norm_efficiency
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_machines[i]

    # Return (index, machine) to explicitly guide the simulator
    if best_machine is not None:
        return (int(best_idx), best_machine)
    else:
        return int(best_idx)
# EVOLVE-BLOCK-END
