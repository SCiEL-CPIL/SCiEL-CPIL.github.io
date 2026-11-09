# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 300.0  # Leave margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    
    Uses a pure Upper Confidence Bound (UCB) strategy.
    
    Structural Change:
    Replaces the hybrid EI/UCB mixture with a pure UCB strategy.
    The Beale function has a wide flat region and a narrow global minimum.
    EI can be zero in flat regions where mu is close to f_best (even if f_best is poor),
    leading to lack of exploration. UCB explicitly rewards uncertainty (sigma),
    which is high in unexplored regions, effectively guiding the search towards
    the narrow basin of attraction at (3, 0.5).
    
    We use a constant beta=2.0 to balance exploration and exploitation.
    This value is chosen based on insights that higher beta values improve
    exploration in the early stages, which is critical for the Beale function.
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Emergency fallback: pure exploitation
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # UCB for minimization: -mu + beta * sigma
    # Higher beta encourages more exploration of high-uncertainty regions
    beta = 2.0
    
    score = -mu + beta * sigma_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
