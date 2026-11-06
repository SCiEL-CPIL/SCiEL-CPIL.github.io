"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next using Lexicographic LPT-Job / SPT-Op heuristic.
    
    Strategy:
    1. Prioritize operations belonging to jobs with the most remaining work (LPT for jobs).
       This ensures that large jobs are kept in the pipeline to reduce the overall makespan.
    2. Among operations with similar remaining work, prioritize those with the shortest
       processing time on their best available machine (SPT for operations).
       This helps in quickly freeing up machines and reducing congestion.
    
    The score is calculated as:
      score = remaining_work * 1000 - min_duration
    
    This effectively creates a lexicographic ordering where remaining_work is the primary
    key and min_duration is the secondary key (minimized).
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Calculate scores
    scores = np.zeros(n, dtype=float)
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        
        # Find the minimum duration among machine options
        # If no options, assume 0 (should not happen in valid open shop)
        if op["machine_options"]:
            min_duration = min(op["machine_options"].values())
        else:
            min_duration = 0.0
            
        # Score: High remaining work is good. Low min_duration is good.
        # Scaling remaining_work by 1000 ensures it dominates the decision.
        scores[i] = remaining_work * 1000.0 - min_duration
        
    # Pick the operation with the highest score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
