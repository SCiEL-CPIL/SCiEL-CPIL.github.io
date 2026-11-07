"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses Upper Confidence Bound (UCB) for a clean exploitation-exploration balance.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # UCB for minimization: prefer low predicted mean (exploitation)
    # and high uncertainty (exploration).
    # beta controls exploration strength. For 2D Branin with ~30 evals,
    # a moderate beta balances finding all 4 local minima vs converging.
    beta = 1.0
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
