"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Pure UCB (Lower Confidence Bound for minimization)
    with adaptive exploration parameter for multimodal optimization.
    
    For minimization, we want to maximize: score = -mu + beta * sigma
    where mu is the posterior mean and sigma is the posterior std.
    
    The exploration parameter beta is adapted based on the relative uncertainty
    of the model. When uncertainty is high relative to the spread of means,
    we explore more (higher beta). When uncertainty is low, we exploit more.
    
    This pure UCB approach is preferred over blended EI+UCB because it
    provides a simpler, more robust exploration-exploitation trade-off
    that works well for multimodal functions like Michalewicz 2D.
    """
    t_start = time.time()
    
    # Ensure proper types and handle numerical issues
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=-1e6)
    sigma = np.maximum(sigma, 1e-12)
    
    n = len(mu)
    
    # Compute relative uncertainty to adapt beta
    sigma_median = np.median(sigma)
    mu_range = np.ptp(mu) if n > 1 else 1.0
    
    if mu_range > 1e-12:
        rel_uncertainty = sigma_median / (mu_range + 1e-12)
    else:
        rel_uncertainty = 1.0
    
    # Adapt beta: base 2.0, scaled by relative uncertainty
    # When rel_uncertainty is high, explore more (beta up to 3.0)
    # When rel_uncertainty is low, exploit more (beta down to 1.0)
    beta = np.clip(2.0 * (0.5 + rel_uncertainty), 1.0, 3.0)
    
    # UCB score for minimization: higher is better
    # We want low mu (good prediction) and high sigma (uncertainty worth exploring)
    score = -mu + beta * sigma
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Safety: check time
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:  # Safety margin before 300s limit
        # Return a simple fallback
        return -mu
    
    return score
# EVOLVE-BLOCK-END
