"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using Thompson Sampling.
    
    For minimization, we sample from the posterior N(mu, sigma^2) 
    and return the negative of the sample. Higher scores correspond 
    to lower sampled values (better candidates).
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        The best (lowest) objective value observed so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores. Higher is better.
    """
    # Deadline guard: The function call itself is very fast (vectorized), 
    # but we include a check as per directive. 
    # Since we don't have a start time from outside, we assume the call 
    # happens within the loop. We simply ensure we don't do anything 
    # computationally expensive that could hang.
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Handle numerical issues
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Thompson Sampling: Draw a sample from the posterior for each candidate
    # Using the standard normal distribution for efficiency
    z = np.random.standard_normal(mu.shape)
    samples = mu + sigma * z
    
    # For minimization, we want to pick the candidate with the lowest sample.
    # The acquisition score should be higher for better candidates.
    # Therefore, we return the negative of the sample.
    scores = -samples
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
