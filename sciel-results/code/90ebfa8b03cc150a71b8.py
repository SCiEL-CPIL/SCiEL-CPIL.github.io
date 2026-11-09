"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement (PoI) per unit cost.
    # PoI is more exploratory than EI and less prone to getting stuck in local optima
    # by focusing on the probability of finding *any* improvement rather than the
    # expected magnitude of improvement.
    
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement
    poi = norm.cdf(z)
    
    # Cost-aware score: higher probability of improvement per unit cost is better
    # We use 1/cost to penalize expensive evaluations
    score = poi / cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
