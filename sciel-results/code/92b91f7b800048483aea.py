"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement with a mild cost penalty.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (N,)
        GP mean predictions at candidate points.
    sigma : np.ndarray, shape (N,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed function value so far.
    cost : np.ndarray, shape (N,)
        Evaluation cost for each candidate point.
    
    Returns:
    --------
    np.ndarray, shape (N,)
        Acquisition values (higher = more promising).
    """
    # Safety margin: return early if we're getting close to the 330s limit
    elapsed = time.time() - _start_time
    if elapsed > 315:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Clip cost to avoid division by zero
    cost = np.clip(cost, 1e-8, None)
    
    # Compute z-score for Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Expected Improvement: E[improvement | improvement > 0]
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = improvement * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: use a mild linear penalty. The cost model is 1 + mean(x_scaled),
    # ranging from 1 to 2. A gentle penalty discourages expensive points slightly
    # without strongly biasing against the global minimum region.
    # Using a smaller slope than before to further reduce bias.
    cost_penalty = 1.0 + 0.1 * (cost - 1.0)
    cost_penalty = np.maximum(cost_penalty, 1e-8)
    
    score = ei / cost_penalty
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
