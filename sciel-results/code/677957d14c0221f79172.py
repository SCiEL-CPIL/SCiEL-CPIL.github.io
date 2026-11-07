"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PI) acquisition function with a small jitter.
    PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    
    This differs from EI by focusing purely on the probability of finding a better
    point, which can lead to more exploratory behavior early on and potentially
    faster convergence in some landscapes.
    """
    # Add a small jitter to sigma to avoid division by zero
    sigma_safe = sigma + 1e-9
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement: Phi(z)
    pi_scores = norm.cdf(z)
    
    # Ensure finite scores
    pi_scores = np.where(np.isfinite(pi_scores), pi_scores, 0.0)
    
    return pi_scores
# EVOLVE-BLOCK-END
