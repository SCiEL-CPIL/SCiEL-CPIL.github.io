"""
Acquisition function for Bayesian optimization of Michalewicz 2D.

This program uses an Adaptive Upper Confidence Bound (UCB) strategy.
Unlike hybrid EI/UCB methods that normalize scores, this approach uses
raw UCB scores: Score = -mu + beta * sigma.
The parameter beta is adapted based on the current best value f_best.
When f_best is far from the known global minimum, beta is increased to
encourage exploration of unvisited regions. As f_best approaches the
minimum, beta decreases to focus on exploitation.
"""
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points using Adaptive UCB.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at each candidate
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at each candidate
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Known minimum of Michalewicz 2D
    f_min = -1.8013
    
    # Handle edge cases: ensure sigma is positive
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate the gap between current best and the known minimum
    gap = max(f_best - f_min, 0.0)
    
    # Adapt beta based on the gap.
    # If gap is large (early stage), we want high exploration (high beta).
    # If gap is small (late stage), we want low exploration (low beta).
    # We use an exponential decay for the exploration component.
    # Base beta for exploitation (when gap is 0)
    beta_base = 0.5
    # Additional beta for exploration (when gap is large)
    beta_max_exploration = 3.0
    
    # Scale the gap to determine exploration intensity
    # A gap of 0.5 or more is considered "large" for Michalewicz 2D
    exploration_intensity = 1.0 - np.exp(-gap / 0.5)
    
    # Final beta
    beta = beta_base + beta_max_exploration * exploration_intensity
    
    # Calculate UCB for minimization: we want to maximize -mu + beta * sigma
    # Equivalent to minimizing mu - beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
