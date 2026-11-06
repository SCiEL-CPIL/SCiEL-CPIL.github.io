"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = 140.0  # Safety margin before 150s kill
_START_TIME = _time.monotonic()


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    Combines the efficiency metric (remaining_work / min_duration) from the best
    inspiration with the composite weighted structure of the current program.
    
    Key insight: the efficiency ratio captures how "urgent" an operation is relative
    to the job's total remaining load. Short operations on jobs with lots of remaining
    work get high priority, which frees machines quickly while addressing long-tail jobs.
    
    Composite score:
    - Primary: efficiency = remaining_work / min_duration (short step relative to total work)
    - Secondary: SPT bonus (1/min_duration) to further favor short operations
    - Tertiary: remaining work (more work = higher priority)
    - Quaternary: fewer remaining ops (closer to completion)
    - Penultimate: machine flexibility (fewer options = more constrained)
    """
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _time.monotonic() - _START_TIME > _DEADLINE:
        return 0
    
    n = len(ready_operations)
    
    scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        durations = list(machine_options.values())
        min_duration = min(durations) if durations else 1.0
        
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Primary: efficiency metric - remaining work per unit of current processing time
        # High efficiency means the current step is short relative to total remaining work
        if min_duration > 0:
            efficiency = remaining_work / min_duration
        else:
            efficiency = remaining_work + 1000.0
        
        # Secondary: SPT bonus - pure shortest processing time preference
        if min_duration > 0:
            spt_score = 1.0 / min_duration
        else:
            spt_score = 1000.0
        
        # Tertiary: remaining work - prioritize jobs with more total work
        work_score = remaining_work
        
        # Quaternary: fewer remaining ops means job is closer to completion
        ops_score = -remaining_ops
        
        # Machine flexibility: operations with fewer machine options are more constrained
        num_machines = len(machine_options)
        flexibility_score = -0.1 * num_machines
        
        # Composite score:
        # Efficiency is the primary driver (captures both SPT and remaining work interaction)
        # SPT provides additional weight to very short operations
        # Remaining work and ops provide tie-breaking and secondary effects
        scores[i] = (
            1.0 * efficiency +          # Primary: efficiency ratio
            2.0 * spt_score +           # Secondary: pure SPT
            0.1 * work_score +          # Tertiary: remaining work
            0.5 * ops_score +           # Quaternary: fewer remaining ops
            flexibility_score            # Penultimate: machine flexibility
        )
    
    # Select the operation with the highest score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
