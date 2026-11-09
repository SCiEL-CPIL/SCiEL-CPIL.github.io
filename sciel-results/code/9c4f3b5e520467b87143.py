"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: ensure we return within a safe time margin
    # The function itself is fast, but we add a check for robustness
    # if the input arrays are unexpectedly huge or malformed.
    # However, shape is fixed to (2000,).
    
    # Ensure inputs are numpy arrays for vectorization
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma and cost are positive to avoid division by zero or negative std
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Probability of Improvement (PI)
    pi = Phi_z
    
    # Composite Score: Probability-weighted EI divided by cost
    # This balances the expected gain with the likelihood of improvement,
    # penalized by the evaluation cost.
    scores = (ei * pi) / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
