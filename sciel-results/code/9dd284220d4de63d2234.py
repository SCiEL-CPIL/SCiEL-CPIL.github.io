# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule using a combined SPT, MWS, and Rigidity (Bottleneck) heuristic.
    
    This heuristic aims to minimize makespan by:
    1. Favoring operations with a short minimum processing time (SPT) to reduce 
       machine occupancy time.
    2. Favoring operations with a large amount of remaining work (MWS) to prioritize 
       critical path jobs.
    3. Favoring operations with fewer machine options (Rigidity) to prioritize 
       bottleneck operations that cannot be easily rescheduled to other machines.
    
    The score for each ready operation is:
        score = min_duration / (remaining_work + 1.0) / max(1, len(machine_options))
    
    We pick the operation with the lowest score.
    """
    if not ready_operations:
        return 0
    
    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    
    # For each operation, find the minimum duration among its machine options
    # and the number of machine options
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else float('inf')
        for op in ready_operations
    ], dtype=float)
    
    option_counts = np.array([
        len(op["machine_options"]) if op["machine_options"] else 1
        for op in ready_operations
    ], dtype=float)
    
    # Ensure remaining_work is at least 1.0 to avoid division by zero
    remaining_works = np.maximum(remaining_works, 1.0)
    
    # Ensure option_counts is at least 1.0
    option_counts = np.maximum(option_counts, 1.0)
    
    # Score: lower is better
    # Division by remaining_work favors higher remaining work (MWS).
    # Multiplication by min_duration favors shorter durations (SPT).
    # Division by option_counts favors fewer options (Rigidity/Bottleneck).
    scores = (min_durations / remaining_works) / option_counts
    
    # Handle any inf values (if no machine options)
    if np.any(np.isinf(scores)):
        # Assign a very high score to operations with no valid machine options
        scores = np.where(np.isinf(scores), 1e9, scores)
        
    # Pick the operation with the lowest score
    best_idx = int(np.argmin(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
