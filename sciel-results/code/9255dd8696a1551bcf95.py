"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next using a composite heuristic.
    
    Heuristics considered:
    1. Most Work Remaining (MWR) - total remaining processing time for the job
    2. Most Operations Remaining (MOR) - number of remaining operations
    3. Shortest Processing Time (SPT) - minimum duration among machine options
    4. Longest Processing Time (LPT) - maximum duration among machine options
    5. Weighted combination of above
    
    We use a weighted composite score that balances:
    - Job urgency (remaining work and ops)
    - Operation complexity (processing time on available machines)
    """
    if not ready_operations:
        return 0
    
    scores = []
    for op in ready_operations:
        # Get remaining work and operations for the job
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        
        # Get machine options and their durations
        machine_options = op.get("machine_options", {})
        if machine_options:
            durations = list(machine_options.values())
            min_duration = min(durations)
            max_duration = max(durations)
            avg_duration = sum(durations) / len(durations)
        else:
            min_duration = 0
            max_duration = 0
            avg_duration = 0
        
        # Composite score: higher is better (we pick max score)
        # Weight remaining work heavily as it captures overall job urgency
        # Add a small bonus for shorter operations to reduce idle time
        # Penalize very long operations slightly to avoid blocking machines
        
        score = (
            10.0 * remaining_work +      # Most work remaining
            5.0 * remaining_ops +         # Most operations remaining
            -0.5 * avg_duration           # Slight preference for shorter ops
        )
        
        scores.append(score)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
