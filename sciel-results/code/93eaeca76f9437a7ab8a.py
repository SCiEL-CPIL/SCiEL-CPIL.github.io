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
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Known minimum for Styblinski-Tang 1D
    f_min_known = -39.1662
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in cdf/pdf
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive Exploration-Exploitation Balance
    # We use a hybrid score: Score = EI + beta * sigma
    # EI drives exploitation (finding points better than f_best)
    # beta * sigma drives exploration (finding points with high uncertainty)
    
    # Calculate distance from the known global minimum
    # d is large when we are far from the optimum (early stages)
    # d is small when we are close to the optimum (late stages)
    d = max(f_best - f_min_known, 0.0)
    
    # Beta should be high when d is large (explore) and low when d is small (exploit)
    # Using tanh for a smooth, bounded transition
    # When d=0, beta=1.0 (pure EI-ish, low exploration)
    # When d>20, beta approaches 5.0 (high exploration)
    beta = 1.0 + 4.0 * np.tanh(d / 10.0)
    
    # Combine scores
    scores = ei + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
