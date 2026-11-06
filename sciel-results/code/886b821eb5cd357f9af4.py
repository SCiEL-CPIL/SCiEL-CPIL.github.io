"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0, 0 # Fallback, shouldn't happen if called correctly

    # 1. Select job: Maximize remaining work (Longest Remaining Processing Time heuristic)
    # This helps keep machines busy with substantial work, reducing idle time at the end.
    remaining_works = np.array([op["remaining_work"] for op in ready_operations])
    best_op_idx = int(np.argmax(remaining_works))
    
    # 2. Select machine: Minimize processing time for the selected operation (Shortest Processing Time)
    # This reduces the blocking time of the machine and allows subsequent jobs to start sooner.
    best_op = ready_operations[best_op_idx]
    machine_options = best_op["machine_options"]
    
    # Find the machine with the minimum duration
    best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
    
    return best_op_idx, best_machine
# EVOLVE-BLOCK-END
