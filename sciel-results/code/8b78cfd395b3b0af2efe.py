"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: track elapsed time and return safe result if needed
    # The evaluator kills after 330s wall-clock. We need to be fast.
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide by cost
    # cost = 1 + mean(x scaled to [0,1]), so cost is in [1, 2]
    # We want higher scores for lower cost
    score = ei / cost
    
    # Add a small UCB exploration term to balance exploitation/exploration
    # UCB = mu + beta * sigma, but we're minimizing, so we want low mu and high uncertainty
    # For minimization with EI, the exploration is already in the sigma*pdf term
    # Let's add a small bonus for high uncertainty when we're not close to optimum
    
    # Normalize f_best to scale the exploration weight
    # When f_best is large (early), explore more; when small, exploit
    f_best_scaled = np.abs(f_best)
    
    # Adaptive beta: larger when f_best is far from 0 (minimum)
    beta = 0.1 * (1 + f_best_scaled)
    
    # UCB term for minimization: we want to explore where sigma is high
    # But EI already does this. Let's keep it simple with just EI/cost
    # The key insight from the inspiration is that cost normalization is stable
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
