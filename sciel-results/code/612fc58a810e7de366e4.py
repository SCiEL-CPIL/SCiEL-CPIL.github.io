"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement (EI) for minimization.
    # EI is generally more robust than LCB as it integrates over the posterior distribution.
    # We normalize by cost to prefer cheaper evaluations.
    
    # Avoid division by zero or negative sigma (though sigma should be >= 0)
    sigma = np.maximum(sigma, 1e-10)
    
    # Calculate the Z-score for the improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Normalize by cost
    # Cost is defined as 1 + mean(x scaled to [0, 1]).
    # We want to maximize EI per unit cost.
    scores = ei / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
