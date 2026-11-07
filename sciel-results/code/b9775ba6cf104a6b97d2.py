"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack.
    
    Strategy:
    Combines global density (total prize / total weight) with bottleneck density
    (total prize / max weight in any constraint) using a geometric mean.
    This rewards items that are efficient overall and specifically efficient in
    their most constraining dimension.
    """
    # Start timer for safety guard (though this function is very fast)
    start_time = time.time()
    
    # Total prize per item (sum across all 5 constraints)
    total_prize = prizes.sum(axis=0)  # shape (100,)
    
    # Total weight per item across all constraints
    total_weight = weights.sum(axis=0)  # shape (100,)
    
    # Max weight per item (bottleneck constraint)
    max_weight = weights.max(axis=0)  # shape (100,)
    
    # Prevent division by zero
    eps = 1e-9
    safe_total_weight = total_weight + eps
    safe_max_weight = max_weight + eps
    
    # Global density: Total prize per unit total weight
    d_global = total_prize / safe_total_weight
    
    # Bottleneck density: Total prize per unit of the heaviest single constraint
    # This highlights items that are "light" in their worst-case constraint
    d_bottleneck = total_prize / safe_max_weight
    
    # Geometric mean of the two densities
    # This balances the global efficiency with the specific bottleneck efficiency
    heuristic = np.sqrt(d_global * d_bottleneck)
    
    # Ensure all values are positive and finite
    # If any value is non-finite or <= 0, replace with a small positive value
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Safety check: if we are taking too long (unlikely for this vectorized op), 
    # we just return the result. The 300s limit is for the whole evaluation, 
    # and this function is called once per instance.
    # No explicit early return needed for the calculation itself as it's O(N).
    
    return heuristic
# EVOLVE-BLOCK-END
