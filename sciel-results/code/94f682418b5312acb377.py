"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Probability of Improvement (PI) for minimization.
    # Score = P(f(x) < f_best) / cost = Phi((f_best - mu) / sigma) / cost
    # This saturates reward for high-confidence improvements and naturally
    # bounds the exploration incentive (max 1/cost per point), avoiding
    # excessive exploration of cheap-but-poor corners with high sigma.
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute the probability of improvement: P(f(x) < f_best)
    # f(x) ~ N(mu, sigma^2), so P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    pi_scores = norm.cdf(z)
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    scores = pi_scores / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
