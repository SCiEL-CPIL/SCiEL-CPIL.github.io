import numpy as np
import time

# Global variable to track start time for deadline guard
_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 15.0  # seconds

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a blend of Expected Improvement (EI) and Uncertainty (sigma) scaled by cost.
    Score = (alpha * EI + (1 - alpha) * beta * sigma) / cost
    """
    # Deadline guard
    current_time = time.time()
    if current_time - _start_time > (TIME_LIMIT - SAFETY_MARGIN):
        # If we are close to the limit, return a safe, finite array.
        # Returning zeros ensures valid output.
        return np.zeros_like(mu)

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)

    # Avoid division by zero in cost
    cost_safe = np.where(cost > 1e-10, cost, 1.0)

    # Parameters
    # alpha: weight for exploitation (EI) vs exploration (sigma)
    # beta: scale for exploration term
    alpha = 0.7
    beta = 1.0

    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the CDF of the standard normal, phi is the PDF
    # We use the standard normal CDF/PDF from scipy.stats.norm
    
    from scipy.stats import norm
    
    z = (f_best - mu) / sigma
    # Handle cases where sigma is very small
    z = np.where(sigma > 1e-10, z, 0.0)
    
    EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    EI = np.maximum(EI, 0.0)
    
    # Exploration term
    exploration = beta * sigma
    
    # Combined score
    raw_scores = alpha * EI + (1 - alpha) * exploration
    
    # Cost-aware scaling
    scores = raw_scores / cost_safe
    
    # Handle potential numerical issues (inf/nan)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
