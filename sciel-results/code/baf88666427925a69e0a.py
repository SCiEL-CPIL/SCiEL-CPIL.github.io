"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with an exploration bonus based on predictive uncertainty.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising to evaluate next).
    """
    # Deadline guard: track elapsed time to avoid timeout
    t_start = time.time()
    deadline = 300.0 - 5.0  # 5 second safety margin
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (EI) with small xi for exploration
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu - xi) / sigma
    xi = 0.01
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by sqrt(cost) for smoother cost scaling
    cost_safe = np.maximum(cost, 1e-8)
    ei_per_cost = ei / np.sqrt(cost_safe)
    
    # Exploration bonus: reward high uncertainty
    # Use percentile-based normalization for robustness
    sigma_95 = np.percentile(sigma_safe, 95)
    if sigma_95 > 0:
        sigma_norm = np.minimum(sigma_safe / sigma_95, 1.0)
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Blend EI with exploration bonus
    # Use adaptive exploration weight based on EI scale
    ei_scale = np.max(ei_per_cost)
    if ei_scale > 0:
        ei_norm = ei_per_cost / ei_scale
    else:
        ei_norm = np.zeros_like(ei_per_cost)
    
    # Adaptive exploration weight: higher when EI is small
    # Use tanh to smoothly transition from high exploration to low exploration
    ei_mean = np.mean(ei_norm)
    exploration_weight = 0.25 * np.tanh(0.5 - ei_mean)
    
    scores = ei_norm + exploration_weight * sigma_norm
    
    # Deadline check: if we're running out of time, return immediately
    if time.time() > deadline:
        # Ensure finite scores
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Final safety: ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
