"""Rewritten program with an adaptive cost-aware acquisition function."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Handle edge cases where sigma is 0 or very small
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement (PI) - useful for exploration
    pi = norm.cdf(z)
    
    # Cost is in [1, 2] range. Use a softer penalty.
    # Instead of dividing by cost, use cost^alpha with alpha < 1
    # This makes the penalty less severe for expensive points
    cost_safe = np.where(cost > 1e-12, cost, 1.0)
    
    # Adaptive weighting: 
    # - When EI is high, trust the exploitation signal
    # - When EI is low, add more exploration via PI and sigma
    # - Scale by inverse cost to prefer cheaper points
    
    # Base score: EI per unit cost (with softened cost)
    cost_soft = cost_safe ** 0.7  # Less aggressive than linear division
    
    # Exploration term: reward high uncertainty, but penalize expensive exploration
    exploration = sigma_safe / cost_soft
    
    # Blend: primarily EI, with a small exploration boost
    # The exploration weight should be small to avoid wasting budget
    w_explore = 0.15
    
    scores = ei / cost_soft + w_explore * exploration
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
