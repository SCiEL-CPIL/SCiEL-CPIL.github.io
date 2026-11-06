"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = 140.0  # Safety margin before 150s kill
_START_TIME = _time.monotonic()


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    Uses a refined heuristic combining:
    - Shortest Processing Time (SPT): prioritize short operations to free machines quickly
    - Remaining Work: prioritize jobs with more remaining work to reduce makespan
    - Remaining Operations: break ties by fewer remaining steps
    - Machine flexibility: prefer operations with fewer machine options (less contention)
    
    The SPT component is weighted heavily because it has the strongest empirical effect
    on job-shop makespan.
    """
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _time.monotonic() - _START_TIME > _DEADLINE:
        return 0
    
    n = len(ready_operations)
    
    # Pre-compute machine option counts for tie-breaking
    scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        durations = list(machine_options.values())
        min_duration = min(durations) if durations else 1.0
        
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # SPT score: shorter processing time = higher priority
        # Use inverse of min_duration to make short ops score higher
        if min_duration > 0:
            spt_score = 1.0 / min_duration
        else:
            spt_score = 1000.0
        
        # Remaining work score: jobs with more work should be prioritized
        # to avoid long-tail effects
        work_score = remaining_work
        
        # Remaining ops: fewer remaining ops means job is closer to completion
        # Prioritize jobs with fewer remaining ops to finish them
        ops_score = -remaining_ops
        
        # Machine flexibility: operations with fewer machine options
        # are more constrained and should be prioritized to avoid blocking
        num_machines = len(machine_options)
        flexibility_score = -0.1 * num_machines
        
        # Composite score with SPT as primary driver
        # SPT has the strongest empirical effect on job-shop makespan
        scores[i] = (
            10.0 * spt_score +       # Primary: shortest processing time
            1.0 * work_score +        # Secondary: remaining work
            0.5 * ops_score +         # Tertiary: fewer remaining ops
            flexibility_score          # Quaternary: machine flexibility
        )
    
    # Select the operation with the highest score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
