"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 2.0  # Seconds to spare

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        # Fallback to pure exploitation (negative mean) if time is critical
        # This ensures we return something valid quickly
        mu = np.asarray(mu)
        scores = -mu
        return np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)

    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Sanitize sigma
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e9, neginf=1e-9)
    
    # Thompson Sampling
    # Draw a sample from the posterior predictive distribution N(mu, sigma^2)
    z = np.random.standard_normal(size=mu.shape)
    samples = mu + sigma * z
    
    # We are minimizing the objective.
    # The acquisition score must be higher for points with lower expected objective.
    # Therefore, we return the negative of the sampled function value.
    scores = -samples
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return scores
# EVOLVE-BLOCK-END
