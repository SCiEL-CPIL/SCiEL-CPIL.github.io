"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI) for minimization
    # Score = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # Higher score means higher probability of improving upon f_best
    
    # Ensure sigma is strictly positive to avoid division by zero
    # The GP noise floor is 1e-6, so sigma should be >= sqrt(1e-6) ~ 0.001
    # But we clamp to a small value just in case
    sigma_safe = np.maximum(sigma, 1e-6)
    
    z = (f_best - mu) / sigma_safe
    scores = norm.cdf(z)
    
    return scores
# EVOLVE-BLOCK-END
