"""
Program using a single, robust Expected Improvement (EI) acquisition function.
This approach is the standard baseline for Bayesian optimization and is highly effective
for low-dimensional problems, avoiding the overhead of complex portfolio selection.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (minimization) acquisition function.
    
    Parameters:
    -----------
    mu : array-like
        Mean of the GP posterior at candidate points.
    sigma : array-like
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
        
    Returns:
    --------
    scores : ndarray
        Expected Improvement scores for each candidate.
    """
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure all scores are finite
    if not np.all(np.isfinite(scores)):
        finite_mask = np.isfinite(scores)
        if np.any(finite_mask):
            mean_val = np.mean(scores[finite_mask])
            scores = np.where(finite_mask, scores, mean_val)
        else:
            scores = np.zeros_like(mu)
            
    return scores
# EVOLVE-BLOCK-END
