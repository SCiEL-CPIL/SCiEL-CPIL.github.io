"""New program: Cost-Aware Expected Improvement (CAEI).
This approach uses the ratio of Expected Improvement to Cost.
This directly optimizes for the most 'bang for the buck' in terms of
objective improvement per unit of evaluation cost.
It is distinct from the previous EHI+LCB hybrid by removing the additive
LCB term and the power-law cost scaling, using a simple reciprocal cost
penalty which is a standard, robust baseline in cost-aware BO."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Aware Expected Improvement (CAEI) acquisition function.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma
    
    # Standard Normal CDF and PDF
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    EI = (f_best - mu) * Phi + sigma * phi
    
    # Cost-Aware Score: EI divided by Cost
    # This favors points that offer high expected improvement relative to their cost.
    # We use a small epsilon in the denominator to prevent division by zero,
    # though costs are >= 1.0 in this problem.
    scores = EI / (cost + 1e-8)
    
    # Handle any numerical issues
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Shift to ensure non-negative scores for stability
    scores = scores - np.min(scores)
    
    return scores
# EVOLVE-BLOCK-END
