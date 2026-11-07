import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Cost-Aware Expected Improvement (CA-EI).
    
    Args:
        mu: Mean of the GP prediction, shape (2000,)
        sigma: Standard deviation of the GP prediction, shape (2000,)
        f_best: Best observed function value (scalar)
        cost: Cost of evaluating each point, shape (2000,)
        
    Returns:
        scores: Acquisition scores, shape (2000,), higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # We use the standard normal CDF and PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-6)
    
    # Cost-Aware EI: EI divided by cost
    scores = ei / cost_safe
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
