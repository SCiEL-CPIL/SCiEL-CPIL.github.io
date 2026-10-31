import math
import time

_start_time = time.time()
TIME_LIMIT = 325.0  # Safety margin before 330s kill

def priority(vector, n):
    # Check time limit
    if time.time() - _start_time > TIME_LIMIT:
        # Return a very low priority to stop adding vectors quickly if we're over time
        # In the greedy loop, this effectively halts the search for high-priority items
        # However, the contract says priority is just a float. The evaluator handles the timeout.
        # We just need to ensure our function doesn't hang. It's O(n) so it's fast.
        pass
    
    # Calculate sum of squares of coordinate values
    # 0^2 = 0, 1^2 = 1, 2^2 = 4
    # This metric heavily favors 2s over 1s and 0s.
    # The strategy is to break the symmetry of the "balanced" heuristic.
    # By prioritizing vectors with more 2s, we are exploring a different region
    # of the {0,1,2}^n space.
    
    sum_sq = 0
    for v in vector:
        sum_sq += v * v
        
    # We want higher priority for higher sum_sq.
    # So we return sum_sq directly, scaled appropriately.
    # To ensure tie-breaking is deterministic and doesn't interfere with the main signal,
    # we can add a small hash component.
    
    # Deterministic hash for tie-breaking
    h = 0
    base = 31
    for val in vector:
        h = (h * base + val) % (10**9 + 7)
        
    # Scale sum_sq to be the dominant term.
    # Max sum_sq is 4*n.
    # Hash is in [0, 1).
    # We want sum_sq to dominate.
    # If we return sum_sq * 1000 + hash, then vectors with higher sum_sq are always preferred.
    
    return float(sum_sq * 1000.0 + h / (10**9 + 7))
