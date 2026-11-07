"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 330.0

def acquisition(mu, sigma, f_best):
    """
    Pure Lower Confidence Bound (LCB) acquisition function.
    
    Strategy:
    - Use LCB for minimization: LCB = mu + k * sigma
    - We want to minimize LCB, so we maximize -LCB = -mu - k * sigma.
    - A high exploration constant (k=3.0) is used to aggressively target
      regions with high posterior variance, leveraging the multimodal nature
      of the Michalewicz function.
    """
    # Time guard: if we are close to the deadline, return a safe fallback
    # This is unlikely to trigger for 2000 points, but satisfies the directive.
    if time.time() > _DEADLINE - 5.0:
        # Fallback: return negative mean (exploitation) to ensure finite output
        return -np.where(np.isfinite(mu), mu, 0.0)

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Fixed high exploration constant
    k = 3.0
    
    # LCB score: higher is better
    # Minimize (mu + k * sigma) -> Maximize (-mu - k * sigma)
    score = -(mu + k * sigma_safe)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
