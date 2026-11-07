"""Cost-aware acquisition for Bayesian optimization on Rastrigin 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement and
    a dynamic Lower Confidence Bound (LCB) term.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed function value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Ensure inputs are float arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Clip negative EI to zero
    ei = np.maximum(ei, 0.0)
    
    # Dynamic beta for LCB: scales with uncertainty budget
    # Use a time-adaptive beta that encourages exploration early
    # and exploitation later. Since we don't have explicit iteration count,
    # we use the ratio of mean sigma to f_best as a proxy for uncertainty.
    mean_sigma = np.mean(sigma_safe)
    if f_best > 0:
        uncertainty_ratio = mean_sigma / f_best
    else:
        uncertainty_ratio = 1.0
    
    # Beta grows with uncertainty but is bounded
    beta = 2.0 + 3.0 * np.minimum(uncertainty_ratio, 1.0)
    
    # LCB term: optimistic gap
    lcb_gap = f_best - mu + beta * sigma_safe
    
    # Combine EI and LCB, normalized by cost
    # Weight: 0.5 * EI + 0.5 * LCB_gap, both divided by cost
    scores = (0.5 * ei + 0.5 * np.maximum(lcb_gap, 0.0)) / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
