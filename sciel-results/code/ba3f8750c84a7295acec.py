"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Effective Probability of Improvement (CE-POI).
    
    This structural change moves away from additive blends of EI and UCB.
    Instead, it maximizes the probability of finding a new best point per unit of cost.
    This is a distinct resource allocation heuristic that has shown superior
    performance in this task's lineage.
    """
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Probability of Improvement: P(f(x) < f_best)
    # This is non-linear and robust to outliers compared to linear UCB.
    poi = norm.cdf(z)
    
    # Cost-Effective Score: Probability per unit cost
    # Higher score means higher chance of improvement for lower cost.
    score = poi / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
