"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule using Average Remaining Operation Duration (AROD).
    
    This heuristic prioritizes operations belonging to jobs with the shortest
    average remaining processing time per operation. This balances the total
    workload with the number of steps, aiming to complete jobs that are
    'light' on average quickly, which can help in reducing the overall
    makespan by clearing the schedule.
    """
    if not ready_operations:
        return 0

    scores = []
    for op in ready_operations:
        rem_work = op["remaining_work"]
        rem_ops = op["remaining_ops"]
        
        # Avoid division by zero, though rem_ops should be >= 1 if op is ready
        if rem_ops == 0:
            avg_dur = 0.0
        else:
            avg_dur = rem_work / rem_ops
            
        scores.append(avg_dur)
    
    # Return the index of the operation with the minimum average duration
    return int(np.argmin(scores))
# EVOLVE-BLOCK-END
