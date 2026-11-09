"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Global deadline guard
_DEADLINE = _time.time() + 145.0  # 5s safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    
    Strategy: For each ready operation, compute a score that balances:
    1. The minimum duration across available machines (shortest processing time on the best machine)
    2. The remaining work of the job (to prioritize jobs with more work left)
    3. A tie-breaker based on the number of remaining operations
    
    We use a weighted combination: primarily minimize the minimum machine duration,
    with a secondary preference for jobs with more remaining work.
    
    Returns: index of the ready operation to start
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Check deadline
    if _time.time() > _DEADLINE:
        return 0
    
    scores = np.empty(n, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        # Get minimum duration across available machines
        machine_options = op["machine_options"]
        if machine_options:
            min_duration = min(machine_options.values())
        else:
            min_duration = 0.0
        
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Score: primarily minimize min_duration, secondarily maximize remaining_work
        # We want to pick the operation with the smallest min_duration,
        # but break ties by preferring more remaining work.
        # Use a weighted score where lower is better:
        #   score = min_duration - small_weight * remaining_work
        # The small weight ensures remaining_work only breaks near-ties in min_duration.
        
        # Scale remaining_work relative to min_duration to avoid dominance
        # Use a small epsilon weight
        weight = 0.01
        scores[i] = min_duration - weight * remaining_work
    
    # Pick the operation with the lowest score
    best_idx = int(np.argmin(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
