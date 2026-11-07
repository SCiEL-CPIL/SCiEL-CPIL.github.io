"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses Variance-Gated Expected Improvement (VEI): standard EI scaled by
    (sigma / sigma_max)^gamma to force exploration of uncertain regions
    and escape local minima in Branin's multimodal landscape.
    
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
    
    # Compute z-score for Expected Improvement (minimization)
    z = (f_best - mu) / sigma_safe
    
    # Standard EI for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Variance gating: scale EI by (sigma / sigma_max)^gamma
    # This penalizes exploitation in low-uncertainty regions
    gamma = 1.5
    sigma_max = np.max(sigma_safe)
    variance_gate = (sigma_safe / (sigma_max + eps)) ** gamma
    
    scores = ei * variance_gate
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
