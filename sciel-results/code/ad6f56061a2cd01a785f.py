import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian Optimization.
    Combines Expected Improvement (EI) with an exploration bonus,
    normalized by evaluation cost.
    
    Args:
    mu: array of posterior means (shape 2000)
    sigma: array of posterior standard deviations (shape 2000)
    f_best: current best observed value (scalar)
    cost: array of evaluation costs (shape 2000)
    
    Returns:
    scores: array of acquisition scores (shape 2000), higher is better.
    """
    start_time = time.time()
    
    # Handle edge cases for numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: reward high uncertainty regions
    # This helps escape local optima in multimodal landscapes like three-hump camel
    # The bonus is scaled by sigma to be adaptive
    exploration_weight = 0.5
    exploration_bonus = exploration_weight * sigma_safe
    
    # Combined score: EI + exploration bonus
    # This balances exploitation (EI) with exploration (uncertainty)
    combined = ei + exploration_bonus
    
    # Normalize by cost to account for varying evaluation costs
    scores = combined / cost_safe
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    # Safety margin check
    elapsed = time.time() - start_time
    if elapsed > 290:
        return scores
        
    return scores
