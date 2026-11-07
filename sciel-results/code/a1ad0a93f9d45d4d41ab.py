import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition for Styblinski-Tang 2D BO.
    Draws a single sample from the GP posterior and scores by -sample/cost.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - sigma: array of shape (2000,), GP std predictions
    - f_best: float, best observed function value
    - cost: array of shape (2000,), cost of evaluating each candidate
    
    Returns:
    - array of shape (2000,), scores (higher = evaluate next)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        mu_safe = np.where(np.isfinite(mu), mu, f_best)
        return -mu_safe / cost_safe
    
    # Ensure inputs are valid
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Thompson Sampling: draw a single sample from the GP posterior
    # sample = mu + sigma * z, where z ~ N(0, 1)
    z = np.random.standard_normal(mu.shape)
    sample = mu + sigma * z
    
    # Cost-aware score: we want to minimize sample/cost
    # So we return negative of that (higher score = better)
    score = -sample / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -1e6)
    
    return score
