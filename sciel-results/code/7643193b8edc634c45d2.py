"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement + uncertainty-driven exploration.

    Parameters
    ----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points (objective units, lower is better).
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points (objective units).
    f_best : float
        Best observed function value so far (lower is better).

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    t0 = time.time()

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)

    # Upper Confidence Bound (UCB) for minimization
    # Score = -mu + beta * sigma
    # We use a beta that balances exploitation and exploration.
    # For Beale's function with a narrow valley, a moderate beta is effective.
    beta = 2.0
    
    # Calculate UCB scores
    scores = -mu + beta * sigma_safe

    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    # Deadline guard (should be fast; this is just a safety net)
    if time.time() - t0 > 325.0:
        pass  # already computed; return immediately

    return scores
# EVOLVE-BLOCK-END
