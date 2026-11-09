"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a combination of:
    1. Expected Improvement (EI) with cost normalization
    2. UCB-like exploration term
    3. Penalization of high-cost regions
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    n = len(mu)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standard EI calculation
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    # cost has shape (2000,) and represents the cost of evaluating each candidate
    # cost = 1 + mean(x scaled to [0, 1]) where x is in [-5, 5]^2
    # So cost is in [1, 2]
    
    # Use a soft cost penalty: instead of hard division, use a weighted combination
    # This avoids issues when cost varies significantly
    
    # Method: EI per unit cost, but with a floor to prevent extreme values
    cost_safe = np.maximum(cost, 1e-6)
    
    # Base score: EI / cost
    base_score = ei / cost_safe
    
    # Add an exploration bonus for high uncertainty regions
    # This helps when EI is low but uncertainty is high
    # Scale exploration by a factor that balances exploitation and exploration
    exploration_weight = 0.3
    exploration = exploration_weight * sigma_safe / cost_safe
    
    # Combine
    score = base_score + exploration
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Clip to reasonable range to avoid numerical issues
    score = np.clip(score, 0.0, 1e6)
    
    return score
# EVOLVE-BLOCK-END
