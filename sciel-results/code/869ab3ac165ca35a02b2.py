import numpy as np
import time

_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def priority(vector, n):
    """
    Prioritize vectors by their geometric position in {0,1,2}^n.
    
    Strategy: 
    1. Center vectors (many 1s) are added first when the set is small and safe.
    2. Penalize vectors that are "balanced" (roughly equal 0,1,2) as they 
       are most likely to form lines with other vectors.
    3. Add a small diversity bonus based on the variance of the coordinate distribution.
    """
    # Check deadline - if we're running out of time, return a simple score
    if time.time() - _start_time > _DEADLINE:
        return sum(vector)
    
    arr = np.array(vector)
    c0 = np.sum(arr == 0)
    c1 = np.sum(arr == 1)
    c2 = np.sum(arr == 2)
    
    # Center score: maximize c1 (vectors near the center of the hypercube)
    center_score = c1 / n
    
    # Balance penalty: vectors with roughly equal 0s, 1s, 2s are most "dangerous"
    # because they can form lines in many directions.
    # The variance of the count distribution indicates how "unbalanced" the vector is.
    counts = np.array([c0, c1, c2])
    balance_variance = np.var(counts) / (n * n + 1e-9)
    
    # Diversity bonus: vectors with higher variance in their coordinate values
    # are more "spread out" and less likely to cluster with other vectors.
    coord_variance = np.var(arr.astype(float))
    
    # Combined score: center-targeting + diversity, minus balance penalty
    score = (100.0 * center_score 
             + 50.0 * coord_variance 
             - 30.0 * (1.0 - balance_variance))
    
    return score
