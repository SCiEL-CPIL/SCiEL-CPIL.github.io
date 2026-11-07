import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling with cost normalization.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP std dev predictions  
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    - scores: array of shape (2000,) - higher means evaluate next
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-8, None)
    
    # Thompson Sampling: draw a single sample from the posterior predictive distribution
    # Use a fixed seed for reproducibility within a single call, but different across calls
    # Actually, we should use random draws for proper Thompson Sampling
    z = np.random.standard_normal(len(mu))
    theta = mu + sigma * z
    
    # Normalized regret: (theta - f_best) / sqrt(cost)
    # Lower is better for the objective, so we want to minimize this
    # For acquisition (higher is better), we negate it
    sqrt_cost = np.sqrt(cost)
    normalized_regret = (theta - f_best) / sqrt_cost
    
    # Higher scores are better, so we negate
    scores = -normalized_regret
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        # Return the best result found so far (all scores are computed)
        return scores
    
    return scores
