"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses Probability of Improvement (PI) with an uncertainty bonus.
    PI measures the probability that a candidate is better than the current best.
    The uncertainty bonus encourages exploration of regions with high variance.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    np.ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Probability of Improvement (PI) for minimization
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Uncertainty bonus
    # Scale sigma to be comparable to PI (which is in [0, 1])
    # Max sigma in the batch
    sigma_max = np.max(sigma_safe)
    if sigma_max > 0:
        sigma_scaled = sigma_safe / sigma_max
    else:
        sigma_scaled = np.zeros_like(sigma_safe)
        
    # Weight for exploration
    # Early on, we want more exploration. Later, less.
    # Since we don't have t, we use a fixed weight.
    # PI is in [0,1]. sigma_scaled is in [0,1].
    weight_exploration = 0.5
    
    scores = pi + weight_exploration * sigma_scaled
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
