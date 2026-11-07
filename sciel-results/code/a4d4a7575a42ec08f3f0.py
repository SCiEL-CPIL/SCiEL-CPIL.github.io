import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-scores
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is positive for logarithm
    # If EI is negative or zero due to numerical issues, clamp to a small positive number
    ei = np.maximum(ei, 1e-12)
    
    # Log-Expected Improvement
    log_ei = np.log(ei)
    
    # Cost-aware score: Log-EI divided by cost
    # Higher score is better
    score = log_ei / cost
    
    # Handle any non-finite values (NaN, Inf) by replacing with a very low score
    score = np.where(np.isfinite(score), score, -1e9)
    
    return score
