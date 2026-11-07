"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points using Probability of Improvement (PI).
    
    PI directly maximizes the probability that a candidate point yields a better objective
    value than the current best, offering a more aggressive exploitation bias than EI.
    For a 1D multi-modal function like Styblinski-Tang, this focused exploitation can
    help converge to the global minimum more reliably.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Probability of Improvement (PI) for minimization
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # where Phi is the standard normal CDF
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Ensure all scores are finite and in [0, 1]
    pi = np.clip(pi, 0.0, 1.0)
    pi = np.where(np.isfinite(pi), pi, 0.0)
    
    return pi
# EVOLVE-BLOCK-END
