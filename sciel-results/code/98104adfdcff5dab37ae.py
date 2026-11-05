"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    Uses a composite heuristic that considers:
    1. Most work remaining (MW) - primary
    2. Shortest minimum duration (to unlock next ops sooner)
    3. Number of remaining ops (critical path consideration)
    4. Machine load balancing (prefer less loaded machines)
    """
    if not ready_operations:
        return 0
    
    if len(ready_operations) == 1:
        return 0
    
    n = len(ready_operations)
    
    # Compute scores for each ready operation
    scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        work = op["remaining_work"]
        durations = list(op["machine_options"].values())
        
        if not durations:
            scores[i] = work
            continue
        
        min_d = min(durations)
        max_d = max(durations)
        avg_d = sum(durations) / len(durations)
        
        # Most work remaining - primary criterion
        score = work
        
        # Prefer operations with shorter minimum duration (faster completion, unlocks next)
        # This is a secondary criterion
        score -= 0.1 * min_d
        
        # Consider the number of remaining operations (longer chains may be more critical)
        remaining_ops = op.get("remaining_ops", 0)
        if remaining_ops > 0:
            # Operations with more remaining work in their job are more critical
            score += 0.05 * remaining_ops
        
        # Penalize operations that have very long maximum duration (bottleneck risk)
        score -= 0.02 * max_d
        
        scores[i] = score
    
    # Add a small random tiebreaker to avoid systematic bias
    # But keep it small enough not to override the heuristic
    rng = np.random.RandomState(int(time * 1000) % 100000)
    tiebreaker = rng.uniform(0, 0.01, size=n)
    scores += tiebreaker
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
