"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 320.0  # seconds, safety margin before 330s limit


def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a simple fallback: just use negative mean
        return -mu
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Handle cases where sigma is very small
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute the standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Compute EI using the standard formula
    # For numerical stability, use scipy or manual computation
    # phi(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
    # Phi(z) = 0.5 * (1 + erf(z/sqrt(2)))
    
    from math import erf, sqrt, pi
    
    # Vectorized computation
    phi_z = np.exp(-0.5 * z**2) / np.sqrt(2 * pi)
    phi_z = np.where(sigma_safe > 1e-10, phi_z, 0.0)
    
    # Phi(z) using erf
    Phi_z = 0.5 * (1 + erf(z / sqrt(2)))
    Phi_z = np.where(sigma_safe > 1e-10, Phi_z, np.where(z > 0, 1.0, 0.0))
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # For points where sigma is very small, EI should be based on whether mu < f_best
    ei = np.where(sigma_safe > 1e-10, ei, np.maximum(f_best - mu, 0.0))
    
    # Add a small exploration bonus to encourage exploration
    # The bonus is proportional to sigma but capped
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma_safe
    
    # Final score: EI + exploration bonus
    scores = ei + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
