"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    
    Uses a combination of Expected Improvement (EI) and a UCB-style exploration term
    to balance exploitation and exploration.
    
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
    # Ensure numerical stability
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative for numerical stability
    ei = np.maximum(ei, 0.0)
    
    # UCB-style exploration term: favors high uncertainty regions
    # The beta parameter controls the exploration-exploitation trade-off
    # For 2D problems with 30 total evaluations, a moderate beta works well
    beta = 1.5
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB with appropriate weighting
    # Normalize both to have similar scale before combining
    ei_scale = np.max(ei) if np.max(ei) > 0 else 1.0
    ucb_scale = np.max(np.abs(ucb)) if np.max(np.abs(ucb)) > 0 else 1.0
    
    # Weighted combination: 70% EI, 30% UCB
    scores = 0.7 * (ei / ei_scale) + 0.3 * (ucb / ucb_scale)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
