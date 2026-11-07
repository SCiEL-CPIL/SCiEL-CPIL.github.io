"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement (PI) weighted by inverse cost
    # PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    
    # Ensure sigma is not zero to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-scores
    z = (f_best - mu) / sigma_safe
    
    # Calculate Probability of Improvement
    pi = norm.cdf(z)
    
    # Weight by a stronger inverse cost function
    # cost is in [1, 2]. 
    # Using 1 / (2*cost - 1) gives a penalty range of [1, 1/3] vs [1, 1/2] for 1/cost.
    # This more aggressively discourages expensive evaluations.
    cost_weight = 1.0 / (2.0 * cost - 1.0)
    
    scores = pi * cost_weight
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
