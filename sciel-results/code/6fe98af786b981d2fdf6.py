"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an open-shop simulation.
    
    Strategy: Use a composite heuristic that balances:
    1. Most work remaining (primary) - to reduce overall makespan
    2. Shortest minimum duration (secondary) - to unlock next operations sooner
    3. Consider machine load/bottleneck awareness
    
    Returns the index of the ready operation to start.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Extract features for each operation
    remaining_works = []
    min_durations = []
    max_durations = []
    num_machines = []
    
    for op in ready_operations:
        work = op["remaining_work"]
        durations = list(op["machine_options"].values())
        if not durations:
            remaining_works.append(0)
            min_durations.append(float('inf'))
            max_durations.append(0)
            num_machines.append(1)
        else:
            remaining_works.append(work)
            min_durations.append(min(durations))
            max_durations.append(max(durations))
            num_machines.append(len(durations))
    
    remaining_works = np.array(remaining_works, dtype=float)
    min_durations = np.array(min_durations, dtype=float)
    max_durations = np.array(max_durations, dtype=float)
    num_machines = np.array(num_machines, dtype=float)
    
    # Handle edge cases
    if np.all(remaining_works == 0):
        return int(np.argmin(min_durations))
    
    # Normalize features to [0, 1] range for weighting
    work_range = remaining_works.max() - remaining_works.min()
    if work_range > 0:
        work_norm = (remaining_works - remaining_works.min()) / work_range
    else:
        work_norm = np.zeros(n)
    
    min_dur_range = min_durations.max() - min_durations.min()
    if min_dur_range > 0:
        min_dur_norm = (min_durations - min_durations.min()) / min_dur_range
    else:
        min_dur_norm = np.zeros(n)
    
    max_dur_range = max_durations.max() - max_durations.min()
    if max_dur_range > 0:
        max_dur_norm = (max_durations - max_durations.min()) / max_dur_range
    else:
        max_dur_norm = np.zeros(n)
    
    # Composite score: prioritize most work, then shorter min duration
    # Higher score = better
    w_work = 1.0
    w_min_dur = 0.15
    w_max_dur = 0.05
    
    # For min_duration, we want shorter to be better, so subtract
    score = (w_work * work_norm - w_min_dur * min_dur_norm - w_max_dur * max_dur_norm)
    
    # Also consider: operations with more machine options might be more flexible
    # but this is a weak signal, so keep weight small
    if np.max(num_machines) > np.min(num_machines):
        num_mach_range = num_machines.max() - num_machines.min()
        num_mach_norm = (num_machines - num_machines.min()) / num_mach_range
        score += 0.02 * num_mach_norm
    
    # Pick the operation with the highest score
    best_idx = int(np.argmax(score))
    
    return best_idx
# EVOLVE-BLOCK-END
