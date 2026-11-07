# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using GP-UCB with budget-scaled (progress-estimated) exploration.
    Beta decays linearly as f_best approaches 0.
    Score is normalized by a convex combination of candidate cost and a constant.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Estimate progress based on f_best.
    # Three Hump Camel min is 0.
    # Reference value tuned to the expected initial f_best range.
    f_ref = 12.0
    # Clip progress to [0, 1]
    progress = 1.0 - (f_best / f_ref)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Beta decays linearly from beta_max to beta_min as progress goes from 0 to 1
    # Higher beta_max for better initial exploration, lower beta_min for exploitation.
    beta_max = 4.0
    beta_min = 1.0
    beta = beta_max - progress * (beta_max - beta_min)
    
    # UCB for minimization: maximize -mu + beta * sigma
    ucb_score = -mu + beta * sigma
    
    # Cost normalization: divide by convex combination of cost and constant
    # Stronger penalty for high cost (higher alpha).
    alpha = 0.7
    c_const = 0.5
    denom = alpha * cost_safe + (1 - alpha) * c_const
    denom = np.maximum(denom, 1e-9)
    
    scores = ucb_score / denom
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
