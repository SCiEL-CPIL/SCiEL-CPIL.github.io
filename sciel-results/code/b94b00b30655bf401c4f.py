"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule: LPT-based heuristic with flexibility and urgency adjustments.
    
    Prioritizes operations with the longest minimum processing time (LPT).
    Breaks ties by preferring operations with fewer machine options (less flexible),
    as they are more constrained. Finally, prefers operations belonging to jobs
    with more remaining work to keep critical jobs progressing.
    """
    if not ready_operations:
        return 0
    
    # Calculate scores for each ready operation
    scores = np.zeros(len(ready_operations))
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        remaining_work = op["remaining_work"]
        
        if not machine_options:
            min_duration = 0
            n_machines = 0
        else:
            durations = list(machine_options.values())
            min_duration = min(durations)
            n_machines = len(durations)
        
        # 1. LPT Component: Primary driver.
        #    We want to maximize this.
        lpt_score = min_duration
        
        # 2. Flexibility Component: Secondary driver.
        #    Less flexible (fewer machine options) should be prioritized.
        #    We subtract n_machines, so lower n_machines yields higher score.
        #    Weighted to be smaller than typical duration differences but significant for ties.
        flexibility_score = -n_machines * 0.1
        
        # 3. Urgency/Workload Component: Tertiary driver.
        #    Jobs with more remaining work are more critical to the makespan.
        #    We add a small bonus for remaining work.
        urgency_score = remaining_work * 0.01
        
        # Combined score
        scores[i] = lpt_score + flexibility_score + urgency_score
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
