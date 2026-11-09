"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    best_score = -float('inf')
    best_idx = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        remaining_work = op["remaining_work"]
        
        # Find the machine with the shortest duration for this operation
        min_duration = float('inf')
        chosen_machine = None
        for m, dur in machine_options.items():
            if dur < min_duration:
                min_duration = dur
                chosen_machine = m
        
        if chosen_machine is None:
            continue

        # Heuristic: Prioritize jobs with more remaining work (Critical Job)
        # and operations with shorter processing time (SPT).
        # Weighting: Remaining work is a strong indicator of criticality.
        score = remaining_work * 10.0 - min_duration
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = chosen_machine

    # Return the index and the specific machine to ensure we use the shortest duration
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
