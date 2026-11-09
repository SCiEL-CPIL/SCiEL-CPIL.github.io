"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for minimizing Griewank 2D using a blend of
    Expected Improvement (EI) and Probability of Improvement (PI), weighted to
    balance exploration and exploitation.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best (lowest) observed value
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    start_time = time.time()
    DEADLINE = 290.0  # Safety margin before 300s limit
    
    # Convert to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # f_best is the best observed value (lowest since we minimize)
    f_best = float(f_best) if np.isfinite(f_best) else 0.0
    
    # Compute Z-score
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    # Compute standard normal PDF and CDF
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * Phi + sigma * phi
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement for minimization
    pi = Phi
    
    # Blend EI and PI: weight PI more when uncertainty is high relative to improvement
    # Use (diff + sigma) as a scale factor for blending
    diff = np.maximum(f_best - mu, 0.0)
    scale = diff + sigma
    # Blend weight: more PI when scale is small (near current best or high uncertainty)
    blend_weight = 0.5 * (1.0 + np.tanh(scale / (sigma + 1e-9)))
    
    # Combined acquisition
    combined = blend_weight * ei + (1.0 - blend_weight) * pi * sigma
    
    # Cost-aware score
    scores = combined / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
