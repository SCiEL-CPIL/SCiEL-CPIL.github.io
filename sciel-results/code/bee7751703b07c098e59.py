"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    
    We maximize EI to select the next point.
    """
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If f_best - mu is negative (mu > f_best), EI should be 0.
    # The formula (f_best - mu) * Phi(z) + sigma * phi(z) naturally handles this?
    # Let's check: if mu > f_best, z is negative. Phi(z) is small. 
    # Actually, the standard EI formula for minimization is:
    # EI = max(0, f_best - mu) * Phi(z) + sigma * phi(z)  -- Wait, no.
    # The correct formula is:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # This formula yields positive values only when there is a chance of improvement?
    # If mu >> f_best, z is large negative. Phi(z) ~ 0, phi(z) ~ 0. EI ~ 0.
    # If mu << f_best, z is large positive. Phi(z) ~ 1, phi(z) ~ 0. EI ~ f_best - mu.
    # This is correct.
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    scores = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure finite scores and non-negative (EI should not be negative)
    scores = np.maximum(scores, 0.0)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
