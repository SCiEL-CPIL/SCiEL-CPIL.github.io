"""Rewritten program for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with a small
    UCB exploration bonus. Both terms are in natural objective units.
    
    Parameters:
    -----------
    mu : array_like, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : array_like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    if len(mu) == 0:
        return np.array([])

    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)

    # Expected Improvement (natural units)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # UCB exploration term: beta * sigma
    # Goldstein-Price range is roughly [3, 12], so scale beta to be modest
    # beta = 1.0 gives exploration comparable to typical EI values early on
    beta = 1.0
    ucb = beta * sigma_safe

    # Combine: EI dominates when improvement is likely, UCB adds exploration
    scores = ei + ucb

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Check deadline
    if time.time() > deadline:
        return scores

    return scores
# EVOLVE-BLOCK-END
