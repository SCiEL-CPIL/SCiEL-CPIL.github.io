import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    Uses EI / cost as the score, which has shown to be optimal for this task.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle sigma = 0 to avoid division by zero
    # If sigma is 0, the distribution is a delta function.
    # EI = max(f_best - mu, 0)
    # Using a small epsilon in denominator for the z-score calculation
    # but we handle the limit case explicitly if needed.
    # For numerical stability in z = (f_best - mu) / sigma:
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI must be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    # Cost is 1 + mean(x), so cost >= 1
    cost_safe = np.maximum(cost, 1e-10)
    scores = ei / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
