import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement with Additive Cost Penalty.
    
    Parameters:
    mu: (2000,) array of GP means in objective units
    sigma: (2000,) array of GP stds in objective units
    f_best: scalar, lowest observed value
    cost: (2000,) array of evaluation costs
    
    Returns:
    (2000,) array of scores, higher is better
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-6)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-6)
    
    # Calculate z-score for Expected Improvement
    # For minimization, improvement is f_best - f(x)
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Use scipy.stats.norm for numerical stability
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative (numerical errors can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Structural Change: Additive Cost Penalty instead of Multiplicative
    # We penalize cost relative to the minimum possible cost (approx 1.0)
    # The penalty weight lambda is scaled by f_best to maintain balance as f_best changes
    # If f_best is 0, no penalty needed (we found the min).
    # If f_best is large, EI is large, so we can tolerate higher cost.
    
    # Heuristic for lambda: 
    # We want the cost penalty to be significant but not dominate EI.
    # Cost range is [1, 2].
    # Let's use a fixed small lambda, or scale with f_best.
    # If f_best is small, EI is small, so cost penalty should be small.
    # lambda = 0.1 * f_best
    # If f_best = 1, lambda = 0.1. Cost penalty max = 0.1 * 1 = 0.1.
    # EI max approx 1. So penalty is 10% of EI.
    # If f_best = 10, lambda = 1. Cost penalty max = 1.
    # EI max approx 10. So penalty is 10% of EI.
    
    # This keeps the relative penalty constant.
    lambda_pen = 0.1 * f_best
    
    # Base cost is 1.0. We penalize extra cost.
    cost_penalty = lambda_pen * (cost - 1.0)
    
    score = ei - cost_penalty
    
    # Handle any non-finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
