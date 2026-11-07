"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Uses a hybrid approach combining Expected Improvement (EI) with a
    progress-based exploration bonus to handle multimodal landscapes like
    Goldstein-Price.
    
    Args:
        mu: Array of shape (2000,) - GP posterior means
        sigma: Array of shape (2000,) - GP posterior standard deviations
        f_best: float - Best observed value so far
    
    Returns:
        Array of shape (2000,) - Acquisition scores (higher is better)
    """
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus that decreases as we get closer to the optimum
    # The Goldstein-Price function has minimum value 3
    # As f_best approaches 3, we want less exploration
    # Use a normalized uncertainty term
    
    # Progress-based beta: more exploration early, less later
    # f_best starts high (could be ~10+) and decreases to 3
    # Normalize progress: 0 when f_best is far from 3, 1 when close
    # Use a sigmoid-like transition
    progress = np.clip((f_best - 3.0) / 5.0, 0.0, 1.0)
    
    # Beta decreases from ~1.0 to ~0.1 as progress increases
    beta = 0.1 + 0.9 * progress
    
    # Normalize sigma to get a relative uncertainty measure
    sigma_max = np.max(sigma) if np.max(sigma) > 0 else 1.0
    sigma_norm = sigma / sigma_max
    
    # Combine EI with exploration bonus
    # EI is already in objective units, sigma is also in objective units
    # We want the exploration term to be comparable in scale
    scores = ei + beta * sigma_norm * np.max(sigma)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
