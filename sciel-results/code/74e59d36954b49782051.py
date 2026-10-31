"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin before 150s kill

# Pre-computed Weibull mode for typical parameters.
# Weibull mode = ( (k-1)/k )^(1/k) for k>1.
# The task says "Weibull sequences of 1000 items with capacity 100".
# Typical Weibull for bin packing: scale=10, shape k~1.5-2.0
# Let's use a reasonable default. If k=1.5, mode = (0.5/1.5)^(1/1.5) = (1/3)^(2/3) ≈ 0.4807
# If k=2.0, mode = (1/2)^(1/2) ≈ 0.7071
# For capacity 100, items are likely scaled. Let's assume scale such that mean ~ 5-10.
# Actually, the prompt says "Weibull sequences of 1000 items with capacity 100".
# Standard Weibull with scale=10, k=1.5 has mean = 10 * Gamma(1+1/1.5) ≈ 10 * 0.9027 ≈ 9.
# Mode = 10 * (0.5/1.5)^(1/1.5) ≈ 10 * 0.4807 ≈ 4.807
# Let's use mode = 4.8 as a heuristic target for remainder.
_WEIBULL_MODE = 4.8

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score is better.
    
    Strategy: Steer remainders toward the Weibull mode to maximize perfect fit likelihood.
    Score = - (remainder - mode)^2
    This is a structural change from Best-Fit (which minimizes remainder) to 
    Mode-Fit (which targets a specific remainder value).
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to simple best-fit if we're running out of time
        remainders = bins - item
        return np.where(remainders >= 0, -remainders, -np.inf)
    
    remainders = bins - item
    # Only consider bins with enough capacity
    valid = remainders >= 0
    
    # Score: negative squared deviation from the target mode
    # We want remainders close to _WEIBULL_MODE
    scores = -(remainders - _WEIBULL_MODE) ** 2
    
    # Set invalid bins to -inf so they are never chosen
    scores[~valid] = -np.inf
    
    return scores
# EVOLVE-BLOCK-END
