import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) divided by cost, with numerical safeguards.
    """
    # Ensure inputs are 1D arrays of the same length
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero and numerical instability
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-score for the standard normal CDF/PDF
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution
    # For numerical stability, use scipy's norm functions
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Ensure cost is positive to avoid division by zero
    cost_safe = np.maximum(cost, 1e-12)
    
    # Cost-aware score: divide by cost
    # This encourages evaluating points that are both promising (high EI) and cheap
    scores = ei / cost_safe
    
    # Handle any non-finite values to ensure validity
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
