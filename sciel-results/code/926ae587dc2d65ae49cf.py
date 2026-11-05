"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule: Most Work Remaining (MWR) job priority, Shortest Processing Time (SPT) machine selection.
    Returns (index, machine).
    """
    if not ready_operations:
        return 0  # Fallback, though ready_operations should not be empty

    best_idx = -1
    best_machine = None
    max_remaining_work = -1
    min_spt = float('inf')

    for i, op in enumerate(ready_operations):
        # Find the machine with the shortest processing time (SPT) for this operation
        min_dur = float('inf')
        chosen_machine = None
        
        # machine_options is a dict: {machine_id: duration}
        for m, d in op["machine_options"].items():
            if d < min_dur:
                min_dur = d
                chosen_machine = m
        
        rw = op["remaining_work"]
        
        # Primary criterion: Maximize remaining work of the job
        # Secondary criterion: Minimize the processing time of the chosen operation (SPT)
        if rw > max_remaining_work:
            max_remaining_work = rw
            best_idx = i
            best_machine = chosen_machine
            min_spt = min_dur
        elif rw == max_remaining_work:
            if min_dur < min_spt:
                best_idx = i
                best_machine = chosen_machine
                min_spt = min_dur

    # If no valid operation found (should not happen), return first
    if best_idx == -1:
        return 0

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
