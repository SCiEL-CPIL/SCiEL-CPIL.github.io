"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PoI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # The CDF of the standard normal distribution
    scores = norm.cdf(z)
    
    # Ensure scores are finite and within [0, 1]
    # norm.cdf already returns values in [0, 1], but numerical errors might occur
    scores = np.clip(scores, 0.0, 1.0)
    
    return scores
# EVOLVE-BLOCK-END
