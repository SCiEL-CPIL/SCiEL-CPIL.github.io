"""Rewritten cost-aware acquisition for shekel 4d BO loop."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Combines Expected Improvement (EI) with a cost-awareness term.
    The cost is 1 + mean(x scaled to [0,1]), so cheaper points (lower mean)
    are favored. We use a multiplicative penalty based on cost to balance
    exploration vs. exploitation with budget efficiency.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (can be slightly negative due to numerical issues)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware weighting:
    # cost ranges from 1 (all zeros) to 2 (all at max, i.e., mean=1 in [0,1])
    # We want to penalize high-cost points.
    # Use a power-based penalty: score = EI^alpha / cost^beta
    # alpha controls exploitation vs exploration
    # beta controls cost sensitivity
    
    # Adaptive: use EI directly but with cost normalization
    # Normalize cost to [0, 1] range for better scaling
    cost_min = 1.0
    cost_max = 2.0
    cost_norm = (cost - cost_min) / (cost_max - cost_min)  # 0 to 1
    
    # Score: combine EI with cost efficiency
    # Higher EI is better, lower cost is better
    # Use: score = EI / (1 + gamma * cost_norm)
    # gamma controls how strongly cost matters
    
    gamma = 1.5  # Cost sensitivity
    
    score = ei / (1.0 + gamma * cost_norm)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
