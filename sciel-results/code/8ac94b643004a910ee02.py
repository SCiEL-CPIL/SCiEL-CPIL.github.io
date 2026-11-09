"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global time tracking for deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 330.0  # Seconds

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we are running out of time, return safe scores immediately
    # The computation below is fast, but this prevents any unexpected delays.
    if time.time() > _TIME_LIMIT - 2.0:
        # Return zeros or a neutral score to avoid crashing
        return np.zeros_like(mu, dtype=float)

    # Ensure sigma is positive and finite to avoid division by zero or negative variance
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Ensure mu is finite; replace non-finite with f_best (neutral)
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    
    # GP-UCB Acquisition Function for Minimization
    # Score = -mu + beta * sigma
    # We maximize this score.
    # beta controls exploration vs exploitation.
    # With only 30 evaluations, strong exploration is beneficial.
    beta = 2.5
    
    scores = -mu_safe + beta * sigma_safe
    
    # Ensure all scores are finite; replace non-finite with 0.0
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
