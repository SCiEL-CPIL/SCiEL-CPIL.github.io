# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule using a look-ahead bottleneck heuristic with machine-centric load balancing.
    
    This heuristic scores each ready operation by considering:
    1. The minimum processing duration (best machine option) - shorter is better
    2. Machine flexibility: operations with fewer machine options are bottlenecks 
       and should be prioritized to avoid delaying them when their limited machines 
       become busy
    3. Remaining work: operations with more remaining work are more urgent
    
    The score for each ready operation is:
        score = min_duration / (num_machine_options * remaining_work)
    
    We pick the operation with the lowest score.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    
    # For each operation, find the minimum duration and count of machine options
    min_durations = np.empty(n, dtype=float)
    num_options = np.empty(n, dtype=float)
    
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            min_durations[i] = min(mo.values())
            num_options[i] = len(mo)
        else:
            min_durations[i] = float('inf')
            num_options[i] = 1.0  # avoid division by zero
    
    # Ensure remaining_work is at least 1.0 to avoid division by zero
    remaining_works = np.maximum(remaining_works, 1.0)
    
    # Score: lower is better
    # min_duration: favors shorter processing times
    # num_options: operations with fewer options (bottlenecks) get lower scores (prioritized)
    # remaining_work: operations with more remaining work get lower scores (prioritized)
    scores = min_durations / (num_options * remaining_works)
    
    # Handle any inf values (if no machine options)
    if np.any(np.isinf(scores)):
        scores = np.where(np.isinf(scores), 1e9, scores)
    
    # Pick the operation with the lowest score
    best_idx = int(np.argmin(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
