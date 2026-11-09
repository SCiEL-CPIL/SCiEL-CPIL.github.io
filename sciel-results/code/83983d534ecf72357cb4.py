"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a blend of Expected Improvement (EI) and Probability of Improvement (PI)
    normalized before cost division, to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value (lowest)
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Guard against zero/negative cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Normalize EI and PI to have comparable scales
    # Use the max of each to normalize, with a small epsilon to avoid division by zero
    ei_max = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    pi_max = np.max(pi) if np.max(pi) > 1e-12 else 1.0
    
    ei_norm = ei / ei_max
    pi_norm = pi / pi_max
    
    # Blend EI and PI with weights that favor exploration early
    # Use a fixed blend: 0.7 * EI + 0.3 * PI
    blend = 0.7 * ei_norm + 0.3 * pi_norm
    
    # Cost-aware: divide by cost
    scores = blend / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
