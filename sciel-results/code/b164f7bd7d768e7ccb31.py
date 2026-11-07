"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI)
    # This is a structural change from Expected Improvement (EI).
    # PI maximizes the probability that the next evaluation improves the current best.
    # It is computationally cheaper and often more robust in low-dimensional spaces.
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    # Calculate the z-score for the improvement
    z = (f_best - mu) / sigma_safe
    
    # The score is the probability that f(x) < f_best
    scores = norm.cdf(z)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
