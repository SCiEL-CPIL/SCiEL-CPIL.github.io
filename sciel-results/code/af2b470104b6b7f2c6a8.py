"""Pure UCB acquisition for Hartmann 6d.

Structural change: Replaces the complex EI/UCB blend and mini-BO refinement
with a pure Upper Confidence Bound (UCB) strategy. Returns -mu - kappa*sigma
where kappa=5.0, aggressively targeting regions of high posterior variance
to maximize the probability of discovering the global minimum in the highly
multimodal Hartmann 6D landscape within the limited 30-evaluation budget.
"""

import numpy as np
import time

# Global deadline guard: ensure we finish within 330s wall-clock
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s safety margin

def acquisition(mu, sigma, f_best):
    """
    Pure UCB acquisition function for minimization.
    
    Parameters:
    mu: (2000,) array of GP mean predictions.
    sigma: (2000,) array of GP standard deviations.
    f_best: float, the best (lowest) observed value so far.
    
    Returns:
    (2000,) array of scores. Higher scores indicate better candidates.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a simple fallback score
        mu = np.asarray(mu, dtype=float)
        return np.where(np.isfinite(mu), -mu, 0.0)
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure finite and non-negative sigma
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Pure UCB: -mu - kappa * sigma
    # Higher score = more promising for minimization
    kappa = 5.0
    scores = -mu - kappa * sigma
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
