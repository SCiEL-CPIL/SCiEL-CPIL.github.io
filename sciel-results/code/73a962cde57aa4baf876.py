"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses Probability of Improvement (PI) for a structurally different
    exploitation-exploration balance compared to EI or UCB.
    PI focuses on the likelihood of finding a point better than f_best,
    rather than the expected magnitude of improvement. This can be
    beneficial for multimodal functions like Branin by preventing
    excessive exploitation of a single local minimum.
    
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
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Variance-Gated EI: scale EI by (sigma / sigma_max)^gamma
    # This penalizes exploitation in low-uncertainty regions, forcing
    # the search to prioritize areas where the GP is still uncertain.
    # Critical for escaping local minima in Branin's multimodal landscape.
    sigma_max = np.max(sigma_safe) + eps
    gamma = 1.5
    variance_gate = (sigma_safe / sigma_max) ** gamma
    
    scores = ei_scores * variance_gate
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
