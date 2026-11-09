"""Rewritten acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with an exploration bonus.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the Gaussian process posterior at candidate points.
    f_best : float
        Best observed value so far (lowest).
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    # Guard against division by zero and ensure finite outputs
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: encourage exploring regions with high uncertainty
    # Scale the exploration bonus relative to the range of EI values
    ei_range = ei.max() - ei.min()
    if ei_range > 1e-12:
        # Normalize EI to [0, 1]
        ei_norm = (ei - ei.min()) / ei_range
        # Exploration term: higher sigma means more exploration
        # Use a tunable weight; 0.1 is a modest exploration bonus
        exploration_weight = 0.1
        scores = ei_norm + exploration_weight * (sigma_safe / (sigma_safe.max() + 1e-12))
    else:
        scores = ei
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
