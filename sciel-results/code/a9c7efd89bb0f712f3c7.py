"""Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PI) acquisition function.
    
    For minimization, the probability of improvement at a point is
    P(f(x) < f_best) = Phi((f_best - mu) / sigma).
    
    We use a small exploration parameter xi to encourage exploration:
    P(f(x) < f_best - xi) = Phi((f_best - xi - mu) / sigma).
    
    To ensure finite scores and avoid degenerate behavior, we add a small
    penalty for very high uncertainty (large sigma) to prevent the optimizer
    from being drawn to regions of pure noise.
    
    Parameters
    ----------
    mu : array-like, shape (2000,)
        Posterior mean at the 2000 candidate points.
    sigma : array-like, shape (2000,)
        Posterior standard deviation at the 2000 candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more desirable to evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Exploration parameter: small value to balance exploitation and exploration
    xi = 0.01 * (f_best - mu.min() + 1e-12) if f_best > mu.min() else 0.0
    
    # Probability of improvement: P(f(x) < f_best - xi)
    z = (f_best - xi - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Add a small bonus for lower mean to break ties and encourage exploitation
    # Scale by the range of mu to make it comparable
    mu_range = mu.max() - mu.min() + 1e-12
    mean_bonus = -mu / mu_range
    
    # Combine PI with a small mean-based bonus
    scores = pi + 0.1 * mean_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
