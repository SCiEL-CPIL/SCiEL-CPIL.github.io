# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    Uses Expected Improvement (EI) with a small exploration bonus.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety checks
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Higher EI means more expected improvement
    
    z = (f_best - mu) / sigma
    
    # Compute Phi(z) and phi(z)
    # Phi(z) = norm.cdf(z)
    # phi(z) = norm.pdf(z)
    
    # To avoid numerical issues with very large/small z, we can use
    # stable computations, but norm.cdf and norm.pdf are generally stable.
    
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Add a small exploration bonus to encourage exploration early on
    # The bonus decreases as f_best approaches 0
    # max_f is 25 (5^2)
    max_f = 25.0
    ratio = np.clip(f_best / max_f, 0.0, 1.0)
    exploration_bonus = 0.1 * np.sqrt(ratio) * sigma
    
    score = ei + exploration_bonus
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
