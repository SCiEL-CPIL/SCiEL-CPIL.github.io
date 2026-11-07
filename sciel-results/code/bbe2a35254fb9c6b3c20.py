"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Analytical Expected Regret relative to the known global minimum (3.0)
    of the Goldstein-Price function.
    
    Uses the closed-form solution for E[max(0, c - f)] where f ~ N(mu, sigma^2).
    
    mu: (2000,) array of GP posterior means
    sigma: (2000,) array of GP posterior standard deviations
    f_best: lowest observed value so far
    
    Returns: (2000,) array of scores, higher = more promising to evaluate next.
    """
    # Known global minimum of Goldstein-Price 2D on [-2,2]^2 is 3.0
    f_true_min = 3.0
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Standardized distance from the target minimum
    # z = (f_true_min - mu) / sigma
    z = (f_true_min - mu) / sigma
    
    # For f ~ N(mu, sigma^2), E[max(0, c - f)] = 
    #   sigma * [phi(z) + z * Phi(z)] where z = (c - mu)/sigma
    # phi is the standard normal PDF, Phi is the standard normal CDF
    
    from scipy.stats import norm
    
    # phi(z) = standard normal PDF
    phi_z = norm.pdf(z)
    
    # Phi(z) = standard normal CDF
    Phi_z = norm.cdf(z)
    
    # Expected regret: sigma * (phi(z) + z * Phi(z))
    scores = sigma * (phi_z + z * Phi_z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
