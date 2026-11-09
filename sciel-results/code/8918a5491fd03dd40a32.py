"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    
    Uses Thompson Sampling: draws a single sample from the posterior at each
    candidate point and returns the negative sample as the score. This provides
    stochastic exploration that adapts naturally to the uncertainty structure.
    
    Args:
        mu: array of shape (2000,) with posterior means in objective units
        sigma: array of shape (2000,) with posterior standard deviations in objective units
        f_best: the lowest observed value so far
    
    Returns:
        array of shape (2000,) with acquisition scores (higher = more promising)
    """
    # Ensure sigma is non-negative and handle any numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Draw a single sample from the posterior at each candidate point
    # For minimization, we want lower values to be better, so we use
    # negative samples as scores (higher score = lower sampled value)
    samples = mu + sigma * np.random.standard_normal(len(mu))
    
    # Return negative samples so that lower objective values get higher scores
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
