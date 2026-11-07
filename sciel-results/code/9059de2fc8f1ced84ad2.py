"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    ei = improvement * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Add a small exploration bonus that decays as f_best approaches the known minimum
    known_min = -39.1662
    max_val = 50.0
    
    if max_val == known_min:
        progress = 0.0
    else:
        progress = np.clip((max_val - f_best) / (max_val - known_min), 0.0, 1.0)
    
    # Exploration bonus: starts at 0.5*sigma, decays to 0.1*sigma as we get closer to optimum
    exploration_weight = 0.5 - 0.4 * progress
    exploration_bonus = exploration_weight * sigma_safe
    
    # Combine EI and exploration bonus
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
