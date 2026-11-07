import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using EI with a cost-adjusted threshold.
    
    Args:
        mu: array of shape (2000,)
        sigma: array of shape (2000,)
        f_best: float, the lowest observed value
        cost: array of shape (2000,), evaluation costs
        
    Returns:
        scores: array of shape (2000,), higher is better
    """
    # Ensure sigma is positive and not zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Cost adjustment parameter
    # We want to favor cheap points.
    # t(x) = f_best + lambda * (c_ref - cost)
    # c_ref = 1.5 (midpoint of [1, 2])
    # If cost < c_ref, t > f_best -> Higher EI
    # If cost > c_ref, t < f_best -> Lower EI
    
    # Let's try a small lambda to gently bias
    # The scale of f is ~10. Cost diff is ~1.
    # lambda = 0.1 seems reasonable.
    # Let's make it slightly stronger to see if it helps.
    # Actually, the seed 1/cost is quite strong.
    # Let's try to combine: EI with adjusted threshold, then divide by cost?
    # No, that might double-penalize.
    # Just use the adjusted threshold EI.
    
    lambda_cost = 0.2
    c_ref = 1.5
    threshold = f_best + lambda_cost * (c_ref - cost)
    
    # Calculate EI with adjusted threshold
    z = (threshold - mu) / sigma
    # Avoid numerical issues
    # norm.cdf(z) for z > 8 is 1.0
    # norm.pdf(z) for z > 8 is 0.0
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei_adjusted = (threshold - mu) * cdf + sigma * pdf
    
    # Ensure finite scores
    # If mu is very high, threshold - mu is negative, cdf is small, pdf is small.
    # EI should be non-negative?
    # Mathematically EI >= 0.
    # But due to floating point, it might be slightly negative.
    ei_adjusted = np.maximum(ei_adjusted, 0.0)
    
    # Optional: Divide by cost again?
    # The threshold adjustment already penalizes cost.
    # Dividing by cost might be too strong.
    # Let's stick with just the adjusted EI.
    
    # Handle NaNs or Infs
    scores = np.where(np.isfinite(ei_adjusted), ei_adjusted, -np.inf)
    
    return scores
