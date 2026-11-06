"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule: select the ready operation that maximizes a composite score
    designed to approximate LPT-like load balancing while accounting for
    remaining work on the job and machine flexibility.
    
    The score combines:
      1. Maximum minimum processing time (max min_duration): prioritizes
         long, inflexible tasks early to prevent late bottlenecks.
      2. Remaining work on the job: favors jobs with more total work left,
         ensuring they are completed before the makespan is determined.
      3. Machine option count penalty: slightly penalizes operations with
         many machine options (more flexible) to prioritize less flexible
         ones that need to be scheduled sooner.
    """
    if not ready_operations:
        return 0
    
    scores = []
    for op in ready_operations:
        machine_options = op["machine_options"]
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        durations = list(machine_options.values())
        if not durations:
            min_duration = 0
            max_duration = 0
        else:
            min_duration = min(durations)
            max_duration = max(durations)
        
        # Primary: max min_duration (LPT-like: long inflexible tasks first)
        # Secondary: remaining work (jobs with more work left should progress)
        # Tertiary: penalize high flexibility (many machine options)
        n_machines = len(machine_options)
        
        # Normalize-ish: use min_duration as primary driver
        # Add a small bonus for remaining work to ensure jobs with lots of
        # remaining ops get progressed
        # Penalize operations that have many machine options (more flexible)
        flexibility_penalty = 0.1 * n_machines
        
        score = min_duration + 0.5 * (remaining_work / max(1, len(durations))) - flexibility_penalty
        
        scores.append(score)
    
    scores = np.array(scores)
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
