import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI and UCB with adaptive exploration.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP std dev predictions  
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    - scores: array of shape (2000,) - higher means evaluate next
    """
    # Ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid division issues
    sigma = np.clip(sigma, 1e-8, None)
    
    # Standard EI
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # UCB component
    # Adaptive beta: start with more exploration, reduce as we have more data
    # We don't know exactly how many iterations we've done, but we can use
    # the spread of sigma as a proxy for uncertainty level
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    
    # If sigma is large overall, we're still exploring -> use higher beta
    # If sigma is small, we're exploiting -> use lower beta
    # Normalize sigma to get a sense of exploration phase
    sigma_norm = sigma / (sigma_max + 1e-8)
    
    # Beta that adapts: higher when uncertainty is high
    beta = 2.0 * (0.5 + 0.5 * np.mean(sigma_norm))
    
    ucb = mu - beta * sigma  # For minimization, we want low mu, high sigma helps
    
    # Combine EI and UCB with cost normalization
    # Use a weighted combination that favors EI when it's strong
    ei_normalized = ei / (np.max(ei) + 1e-8)
    ucb_normalized = (ucb - np.min(ucb)) / (np.max(ucb) - np.min(ucb) + 1e-8)
    
    # Blend: 70% EI, 30% UCB
    combined = 0.7 * ei_normalized + 0.3 * ucb_normalized
    
    # Cost-aware: divide by cost
    scores = combined / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
