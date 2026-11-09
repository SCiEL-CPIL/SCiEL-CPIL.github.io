"""
Acquisition function for Bayesian optimization of Michalewicz 2D.

This program uses a Thompson Sampling approach.
At each step, we draw a sample from the GP posterior for each candidate.
The candidate with the lowest sampled value is selected.
This naturally balances exploration and exploitation:
- High uncertainty (sigma) leads to a wider distribution, increasing the
  probability of drawing a very low value.
- Low mean (mu) shifts the distribution towards lower values.

For minimization, we want the lowest value.
We return scores = -sampled_value, so higher scores correspond to lower values.
"""
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points using Thompson Sampling.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at each candidate
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at each candidate
    f_best : float
        Best (lowest) observed function value so far (not directly used in TS,
        but kept for signature compatibility)
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Draw samples from N(mu, sigma^2)
    # We use a fixed seed for reproducibility in testing, but in a real loop,
    # we would want random samples. However, the acquisition function is called
    # once per iteration. We should use a random state to ensure different
    # samples are drawn in different calls if possible, or just use np.random.
    # Since we don't have access to a global RNG state that persists between calls
    # in a guaranteed way (unless we use np.random.seed, which is global),
    # we will just use np.random.normal.
    
    # To avoid issues with the global random state being reset or not,
    # we just draw samples.
    samples = np.random.normal(mu, sigma_safe)
    
    # We want to minimize the function value.
    # So we assign higher scores to lower samples.
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
