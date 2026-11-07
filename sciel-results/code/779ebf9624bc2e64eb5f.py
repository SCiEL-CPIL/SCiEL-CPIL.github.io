"""Rewritten with Expected Improvement + UCB hybrid acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    n = len(mu)
    
    # Clamp sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI) for minimization:
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Compute z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    # phi(z) = standard normal PDF
    phi_z = norm.pdf(z)
    # Phi(z) = standard normal CDF
    Phi_z = norm.cdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Add a small UCB term to encourage exploration in uncertain regions
    # UCB = kappa * sigma
    kappa = 0.5
    ucb = kappa * sigma_safe
    
    # Combine EI and UCB: higher score means evaluate next
    # EI is already higher when more promising, UCB adds exploration
    scores = ei + ucb
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
