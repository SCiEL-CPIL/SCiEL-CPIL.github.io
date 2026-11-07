"""Acquisition function for Bayesian optimization of Weierstrass 1D using Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points using Thompson Sampling.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the Gaussian process posterior at candidate points.
    f_best : float
        Best (minimum) objective value observed so far. (Not used directly in TS, 
        but kept for interface compatibility).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate point. Higher scores indicate 
        more promising candidates to evaluate next. 
        Note: Since we are minimizing, we return negative samples so that 
        'higher' scores correspond to 'lower' predicted function values.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases
    if not np.isfinite(f_best):
        # If f_best is invalid, fall back to random exploration
        return np.random.randn(len(mu))
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw samples from the posterior: N(mu, sigma)
    # We use a single draw per candidate point
    # np.random.normal(loc, scale)
    samples = np.random.normal(loc=mu, scale=sigma_safe)
    
    # Since we want to minimize the function, lower values are better.
    # The acquisition function should return higher scores for better candidates.
    # Therefore, we negate the samples.
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
