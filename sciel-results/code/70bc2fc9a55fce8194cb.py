"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    EI(x) = E[max(f_best - f(x), 0)]
    
    This function balances exploration and exploitation by considering the
    expected magnitude of improvement over the current best observed value.
    It is generally more effective than Probability of Improvement (PI)
    because it rewards points that are not just likely to be better, but
    significantly better.
    """
    # Add a small jitter to sigma to avoid division by zero
    # Use maximum to ensure sigma is at least 1e-9, handling negative sigmas if any (though unlikely)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute standard normal PDF and CDF
    pdf_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    # Calculate Expected Improvement
    # Formula: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei_scores = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure finite scores
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    return ei_scores
# EVOLVE-BLOCK-END
