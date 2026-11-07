"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for minimization.
    
    For each candidate x_i with posterior predictive N(mu_i, sigma_i^2),
    we draw a single sample s_i ~ N(mu_i, sigma_i^2) and use -s_i as the
    acquisition score (higher means more promising for minimization).
    
    This naturally balances exploration and exploitation: candidates with
    high uncertainty (large sigma) have a wider distribution of possible
    samples, so they can occasionally produce very low (good) objective
    values, while candidates with low uncertainty will consistently produce
    scores close to their mean.
    """
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Draw a single sample from the predictive posterior for each candidate
    # Using a fixed random state would make this deterministic, but Thompson
    # Sampling typically uses fresh randomness each call.
    # We use np.random.default_rng() for better random number generation.
    rng = np.random.default_rng()
    
    # Sample from N(mu, sigma^2)
    samples = rng.normal(mu, sigma_safe)
    
    # For minimization, we want to select the candidate with the lowest
    # objective value. So the acquisition score is the negative of the sample.
    scores = -samples
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
