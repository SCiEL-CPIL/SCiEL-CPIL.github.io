"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Pure Thompson Sampling acquisition for Bayesian optimization.
    
    Samples a single function value from the predictive distribution N(mu, sigma^2)
    at each candidate and returns the negative sampled values as scores.
    The candidate with the highest score (lowest sampled value) is selected next.
    
    Parameters:
    -----------
    mu : array-like, shape (n_candidates,)
        Posterior mean predictions at candidate points.
    sigma : array-like, shape (n_candidates,)
        Posterior standard deviations at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        Acquisition scores; higher means more desirable to evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = len(mu)
    
    if n == 0:
        return np.array([])
    
    t_start = time.time()
    deadline = t_start + 280.0
    
    # Ensure sigma is positive
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Pure Thompson Sampling: draw one sample from the predictive distribution
    # at each candidate point. The score is the negative of the sample,
    # so the candidate with the lowest sampled value gets the highest score.
    rng = np.random.default_rng()
    samples = rng.normal(loc=mu, scale=sigma_safe, size=n)
    
    scores = -samples
    
    # Check deadline
    if time.time() > deadline:
        scores = np.where(np.isfinite(scores), scores, -1e10)
        return scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
