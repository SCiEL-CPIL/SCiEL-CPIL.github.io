"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import ndtr, ndtri
import math

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s kill

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Probability of Improvement (PI) 
    and Expected Improvement (EI).
    
    Strategy:
    - PI (Probability of Improvement) is highly exploitative, favoring points 
      with the highest chance of beating the current best.
    - EI (Expected Improvement) balances exploitation and exploration.
    - A hybrid with 70% weight on PI and 30% on normalized EI accelerates 
      convergence on unimodal functions like the 1D sphere by reducing 
      unnecessary exploration in high-variance regions while retaining 
      enough exploration signal to avoid local minima traps.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to a simple, fast, safe score if time is critical
        # Returning EI is a reasonable fallback as it's a standard baseline
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        phi_z = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
        ei = (f_best - mu) * ndtr(z) + sigma_safe * phi_z
        return np.where(np.isfinite(ei), ei, 0.0)

    # Numerical safety for sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute standard normal score z
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute PI (Probability of Improvement)
    # PI = P(f(x) < f_best) = Phi(z)
    pi_score = ndtr(z)
    
    # Compute EI (Expected Improvement)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # phi(z) = exp(-0.5 * z^2) / sqrt(2 * pi)
    phi_z = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    ei_score = (f_best - mu) * pi_score + sigma_safe * phi_z
    
    # Normalize EI to be comparable to PI (which is in [0, 1])
    # We divide by the max EI in the batch to scale it to [0, 1]
    # If max EI is 0 or negative, EI scores are all 0 or negative, 
    # but EI is always >= 0 for minimization if f_best is the min seen so far.
    # If all EI are 0, the hybrid will just be PI.
    max_ei = np.max(ei_score)
    if max_ei > 1e-12:
        ei_norm = ei_score / max_ei
    else:
        ei_norm = np.zeros_like(ei_score)
        
    # Hybrid score: 70% PI, 30% Normalized EI
    # This biases towards exploitation (PI) while keeping some exploration (EI)
    score = 0.7 * pi_score + 0.3 * ei_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
