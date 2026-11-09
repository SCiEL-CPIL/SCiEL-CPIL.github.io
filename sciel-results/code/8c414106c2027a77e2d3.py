import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Upper Confidence Bound (UCB) for minimization
    # Score to maximize: -mu(x) + beta * sigma(x)
    # This balances exploitation (low mu) and exploration (high sigma).
    
    # Beta controls the exploration-exploitation trade-off.
    # A fixed beta is used as the iteration count is not available.
    beta = 1.0
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 0.0)
    
    # UCB score (higher is better)
    ucb_score = -mu + beta * sigma_safe
    
    # Cost-aware score: maximize UCB per unit cost
    score = ucb_score / cost
    
    # Mask invalid values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
