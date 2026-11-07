"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition with diversity penalty.
    
    Combines cost-normalized EI with a diversity term that rewards
    exploring regions far from already-well-characterized areas (low sigma).
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value (lower is better).
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety margins
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # z-score for EI computation
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -20.0, 20.0)
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Scale EI to be comparable across different problem scales
    # Use the spread of mu to normalize
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = max(mu_max - mu_min, 1e-10)
    
    # Normalize EI by the typical improvement scale
    ei_normalized = ei / (mu_range + 1e-10)
    
    # Diversity-aware exploration: reward high sigma (uncertainty)
    # but penalize being in regions that are already well-explored
    # The sigma itself serves as a proxy for distance to evaluated points
    # with low sigma near evaluated points and high sigma far away
    
    # Exploration term: sigma normalized to be comparable
    sigma_normalized = sigma_safe / (np.max(sigma_safe) + 1e-10)
    
    # Combined acquisition: weighted combination of exploitation and exploration
    # Weight exploration more heavily early on (when f_best is high relative to mu range)
    # and exploitation more as we converge
    f_best_norm = (f_best - mu_min) / (mu_range + 1e-10)
    f_best_norm = np.clip(f_best_norm, 0.0, 1.0)
    
    # As f_best approaches the minimum, reduce exploration weight
    # When f_best_norm is high (far from optimum), explore more
    # When f_best_norm is low (near optimum), exploit more
    exploration_weight = 0.3 + 0.4 * f_best_norm
    
    combined = ei_normalized + exploration_weight * sigma_normalized
    
    # Cost-aware: divide by cost
    score = combined / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    # Deadline guard: if we've been running too long, return immediately
    elapsed = time.time() - t0
    if elapsed > 280.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
