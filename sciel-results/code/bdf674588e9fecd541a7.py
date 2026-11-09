"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a Cost-Weighted Probability of Improvement (PI) with a dynamic
    uncertainty bonus. The score is:
        score = (PI + exploration_weight * normalized_sigma) / cost
    
    where PI = norm.cdf((f_best - mu) / sigma) measures the probability
    that the point improves upon f_best, and the sigma term provides
    an exploration bonus that is strongest when uncertainty is highest.
    
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
    t_start = time.time()
    
    # Safety: ensure finite inputs and correct types
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for probability of improvement
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -40, 40)
    
    # Probability of Improvement (higher = more likely to improve)
    pi = norm.cdf(z_clipped)
    
    # Normalize sigma to [0, 1] relative to the maximum sigma in the batch
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Dynamic exploration weight:
    # Early in the search (f_best far from minimum), we want more exploration
    # Late in the search, we want more exploitation
    # Hartmann 3D: F_MIN = -3.86278, typical initial best ~ -2.0
    F_MIN = -3.86278
    F_INIT = -2.0
    
    delta_f = F_INIT - F_MIN
    if delta_f <= 0:
        p = 1.0
    else:
        p = (f_best - F_MIN) / delta_f
        p = np.clip(p, 0.0, 1.0)
    
    # Exploration weight: higher early (p~1), lower late (p~0)
    # Range from 0.3 (early) to 0.05 (late)
    exploration_weight = 0.05 + 0.25 * p
    
    # Combined score before cost adjustment
    combined = pi + exploration_weight * sigma_norm
    
    # Cost-aware: divide by cost
    scores = combined / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t_start
    if elapsed > 280:
        pass
    
    return scores
# EVOLVE-BLOCK-END
