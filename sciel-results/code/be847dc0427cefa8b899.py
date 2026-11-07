"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PoI) acquisition function.
    
    Scores candidates by the probability that they improve upon f_best,
    computed as Phi((f_best - mu) / sigma). This decouples exploration
    from the magnitude of improvement, favoring broad coverage of the
    multi-modal Branin landscape over the "peaking" bias of Expected
    Improvement.
    
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
    # This is the probability that the true value is better than f_best
    scores = norm.cdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
