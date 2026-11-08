import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Modified Expected Improvement (EI) with an exploration parameter xi,
    normalized by the square root of the evaluation cost.
    
    This strategy was found to outperform hybrid EI+UCB approaches for the
    Ackley 2D function in previous evaluations.
    
    Parameters:
    mu (np.ndarray): Mean predictions from the GP, shape (N,)
    sigma (np.ndarray): Standard deviation predictions from the GP, shape (N,)
    f_best (float): The best (lowest) observed objective value so far.
    cost (np.ndarray): Cost of evaluating each point, shape (N,)
    
    Returns:
    np.ndarray: Acquisition scores, higher is better, shape (N,)
    """
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: return simple EI-based scores to finish quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.maximum(ei, 0.0) / cost_safe

    # Numerical stability
    eps = 1e-9
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Exploration parameter xi.
    # xi > 0 encourages exploring regions with high uncertainty even if 
    # the mean is not significantly better than f_best.
    # Ackley is multimodal, so some exploration is beneficial.
    # 0.1 was the parameter in the best-performing variant (score 1.4865).
    xi = 0.1
    
    # Calculate z-score for modified EI
    # z = (f_best - mu - xi * sigma) / sigma
    z = (f_best - mu - xi * sigma_safe) / sigma_safe
    
    # Calculate CDF and PDF of standard normal distribution
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement with xi
    # EI = (f_best - mu - xi*sigma) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu - xi * sigma_safe) * cdf + sigma_safe * pdf
    
    # EI should be non-negative, but numerical issues can cause tiny negatives
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring
    # We divide by cost^gamma. 
    # gamma = 0.5 (square root) was used in the best-performing variant.
    # This balances the cost penalty without suppressing exploration too much.
    gamma = 0.5
    scores = ei / np.power(cost_safe, gamma)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
