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
    
    # Weight by inverse cost
    # cost is guaranteed to be > 0 (min 1.0)
    scores = pi / cost
    
    # Ensure all scores are finite
    # If any NaN or Inf occur, replace with 0 (or a very small number)
    # However, the problem states "Return 2000 finite scores... anything else invalidates"
    # So we must guarantee finiteness.
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
