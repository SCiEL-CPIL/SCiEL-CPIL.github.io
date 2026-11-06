"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    Higher score = preferred bin.
    
    Strategy: Smooth best-fit surrogate with adaptive fragment penalty.
    
    We use a smooth scoring function that:
    1. Primarily minimizes remaining capacity (best-fit behavior)
    2. Uses a sublinear penalty (power < 1) to reduce sensitivity to large remainders
    3. Applies a smooth penalty for very small remainders (fragments)
       using a smooth step function instead of a hard threshold
    
    The score is: score = -(rem + eps)^p - smooth_fragment_penalty(rem)
    
    Where:
    - p = 0.6 (slightly more aggressive than sqrt, less than linear)
    - smooth_fragment_penalty uses a smoothstep-like function around rem < threshold
    - This avoids the kink in the hard-threshold fragment penalty
    
    For Weibull distributions with capacity 100, this balances:
    - Filling bins efficiently (minimize bins used)
    - Avoiding tiny unusable fragments
    - Smoothness for better generalization
    """
    t0 = time.time()
    
    # Bins that can fit the item
    remaining = bins - item
    valid = remaining >= 0
    
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if np.any(valid):
        rem = remaining[valid]
        eps = 1e-9
        
        # Primary term: sublinear best-fit
        # p=0.6 is slightly more aggressive than sqrt (0.5), less than linear (1.0)
        p = 0.6
        base_score = -(rem + eps) ** p
        
        # Smooth fragment penalty: penalize remainders below a threshold
        # Use a smooth transition: penalty = A * smoothstep(threshold, rem)
        # smoothstep goes from 1 at rem=0 to 0 at rem=threshold
        # We want penalty to be HIGH when rem is small, LOW when rem is large
        # So: penalty = A * (1 - smoothstep(0, threshold, rem))
        # smoothstep(0, t, x) = 3x^2/t^2 - 2x^3/t^3 for x in [0,t], 0 for x<0, 1 for x>t
        # We want: penalty = A * (1 - smoothstep(0, threshold, rem))
        #          = A * (1 - (3*(rem/threshold)^2 - 2*(rem/threshold)^3)) for rem < threshold
        #          = A * (1 - 3*(rem/threshold)^2 + 2*(rem/threshold)^3)
        # At rem=0: penalty = A
        # At rem=threshold: penalty = 0
        
        threshold = 5.0  # Remainders below this get penalized
        A = 0.3  # Penalty strength
        
        # Compute smooth fragment penalty
        x = np.clip(rem / threshold, 0.0, 1.0)
        smoothstep_val = 3.0 * x**2 - 2.0 * x**3
        fragment_penalty = A * (1.0 - smoothstep_val)
        
        scores[valid] = base_score - fragment_penalty
    
    # Deadline guard
    if time.time() - t0 > 110:
        # Fallback: simple best-fit
        scores = np.where(valid, -remaining, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
