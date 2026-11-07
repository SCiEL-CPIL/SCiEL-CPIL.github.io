"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses Probability of Improvement (PI) with a small uncertainty bonus.
    
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
    
    # Compute z-score for Probability of Improvement
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Compute standard normal CDF
    pi_scores = norm.cdf(z)
    
    # Add a small exploration bonus proportional to sigma
    # This helps escape local minima by favoring high uncertainty regions
    # when PI is similar, but PI remains the primary driver.
    exploration_weight = 0.1
    scores = pi_scores + exploration_weight * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
