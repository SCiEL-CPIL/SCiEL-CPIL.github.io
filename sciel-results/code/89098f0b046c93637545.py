"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Enhanced Probability of Improvement (PoI) acquisition function.
    
    Combines PoI with a small exploration bonus to avoid pure exploitation
    while maintaining the broad-coverage properties of PoI.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions at candidate points.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed value so far (lowest).
    
    Returns
    -------
    scores : np.ndarray, shape (2000,)
        Finite scores; higher means evaluate next.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Normalized improvement z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: Phi(z)
    poi = norm.cdf(z)
    
    # Add a significant exploration bonus weighted by uncertainty
    # This helps explore regions with high uncertainty to avoid local optima
    # in the multi-modal Branin landscape
    exploration_weight = 2.0
    exploration_bonus = exploration_weight * sigma / (sigma + 1.0)
    
    scores = poi + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
