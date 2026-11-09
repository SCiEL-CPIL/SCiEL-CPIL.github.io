"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses Upper Confidence Bound (UCB) for minimization.
    Score = -mu + beta * sigma.
    A fixed beta of 2.0 is used to balance exploration and exploitation,
    which has shown superior performance in high-dimensional spaces.
    
    Args:
        mu: Array of predicted means (shape (2000,)).
        sigma: Array of predicted standard deviations (shape (2000,)).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores (shape (2000,)), higher is better.
    """
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.nan_to_num(sigma, nan=1e-8, posinf=1e8, neginf=1e-8)
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e8, neginf=-1e8)
    
    # UCB parameter
    beta = 2.0
    
    # Compute UCB score for minimization
    # We want to minimize f, so we maximize the score: -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Clean up non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e8, neginf=-1e8)
    
    return scores
# EVOLVE-BLOCK-END
