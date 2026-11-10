"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s hard kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement
    and Probability of Improvement, normalized by cost.
    
    This approach combines the exploration-exploitation balance of EI
    with the bounded stability of PoI, then normalizes by evaluation cost
    to create a cost-effective improvement metric.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far (lower is better).
    cost : np.ndarray, shape (2000,)
        Evaluation cost for each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = better to evaluate next).
    """
    # Time guard: if we're running low on time, return a safe fallback
    if time.time() - _start_time > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-12)
        cost_safe = np.maximum(cost, 1e-12)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -50, 50)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Numerical stability
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Compute z-score for improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50, 50)
    
    # Expected Improvement for minimization
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement for minimization
    poi = cdf_z  # P(f(x) < f_best)
    
    # Hybrid: combine EI and PoI with a weighting
    # EI captures magnitude of expected improvement
    # PoI captures probability of any improvement
    # Weight them equally for balanced exploration-exploitation
    hybrid = 0.5 * ei + 0.5 * poi
    
    # Cost normalization: divide by cost to get improvement per unit cost
    # This creates a "cost-effective improvement" metric
    score = hybrid / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
