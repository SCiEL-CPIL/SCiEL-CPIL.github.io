"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    scores = []
    for op in ready_operations:
        remaining_work = op["remaining_work"]
        machine_options = op["machine_options"]
        
        if not machine_options:
            scores.append(0)
            continue
            
        # Calculate the minimum processing time among available machines
        min_duration = min(machine_options.values())
        
        # Handle division by zero just in case, though durations should be > 0
        if min_duration <= 0:
            min_duration = 1e-9
            
        # Number of machine options (flexibility)
        num_options = len(machine_options)
        
        # Heuristic:
        # 1. Prioritize jobs with more remaining work (critical path length).
        # 2. Prioritize operations with shorter processing times (SPT effect, clears job faster).
        # 3. Prioritize operations with fewer machine options (rigidity, less flexibility to delay).
        
        # Score = (Remaining Work / Min Duration) * (1 + 1/Num Options)
        # High remaining work / low duration = High urgency.
        # Low num options = High multiplier (rigid).
        
        base_score = remaining_work / min_duration
        flexibility_factor = 1.0 + (1.0 / num_options)
        
        scores.append(base_score * flexibility_factor)

    # Pick the operation with the highest score
    # If there are ties, argmax picks the first one, which is stable enough.
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
