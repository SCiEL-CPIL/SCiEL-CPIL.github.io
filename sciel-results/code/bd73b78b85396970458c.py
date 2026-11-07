"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    Uses a dynamic UCB-style acquisition that balances exploration and exploitation.
    
    The beta parameter starts high when f_best is large (early search, more exploration)
    and decreases as f_best approaches 0 (late search, more exploitation).
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Dynamic beta: high when f_best is large, low when f_best is near 0
    # f_best for Ackley starts around ~1-3 and goes to 0
    # Use a smooth transition
    # When f_best is large (>2), beta ~ 2.5 (exploration)
    # When f_best is small (<0.1), beta ~ 0.5 (exploitation)
    # Linear interpolation in log-space or direct
    
    # Scale: map f_best from [0, 5] to beta [0.5, 2.5]
    # Clamp f_best to [0, 5]
    f_best_clamped = np.clip(f_best, 0.0, 5.0)
    
    # beta = 0.5 + 2.0 * (1 - f_best_clamped / 5.0)
    # When f_best=5: beta = 0.5
    # When f_best=0: beta = 2.5
    # Wait, we want high beta (exploration) when f_best is large
    # So beta should be high when f_best is large
    # beta = 0.5 + 2.0 * (f_best_clamped / 5.0)
    # When f_best=0: beta = 0.5 (exploitation, we're close to optimum)
    # When f_best=5: beta = 2.5 (exploration, we're far from optimum)
    
    beta = 0.5 + 2.0 * (f_best_clamped / 5.0)
    
    # UCB: mu + beta * sigma
    # Higher values mean more promising
    scores = mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
