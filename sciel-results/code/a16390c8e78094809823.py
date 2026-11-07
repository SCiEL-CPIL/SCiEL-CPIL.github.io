# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Diversity-aware acquisition function for cost-aware Bayesian optimization.
    
    Combines cost-normalized Expected Improvement with a crowding penalty
    based on GP uncertainty to penalize redundant evaluations in dense clusters.
    
    Args:
        mu: array of shape (2000,), GP mean predictions
        sigma: array of shape (2000,), GP standard deviation predictions
        f_best: scalar, best observed value so far
        cost: array of shape (2000,), evaluation cost for each candidate
    
    Returns:
        array of shape (2000,), acquisition scores (higher = evaluate next)
    """
    # Compute Expected Improvement
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-normalized EI
    cost_safe = np.maximum(cost, 1e-9)
    ei_cost = ei / cost_safe
    
    # Diversity-aware crowding penalty using GP uncertainty structure:
    # Low sigma indicates well-explored (crowded) regions.
    # We use a smooth, adaptive penalty that scales with the local EI landscape.
    
    # Use percentiles for robust normalization
    sigma_p1 = np.percentile(sigma, 1)
    sigma_p99 = np.percentile(sigma, 99)
    sigma_range = max(sigma_p99 - sigma_p1, 1e-9)
    
    # Normalize sigma to [0, 1]
    sigma_norm = (sigma - sigma_p1) / sigma_range
    sigma_norm = np.clip(sigma_norm, 0.0, 1.0)
    
    # Crowding penalty: stronger in low-sigma (well-explored) regions
    # Use exponential decay for smoother gradient
    crowding_penalty = np.exp(-3.0 * sigma_norm)
    
    # Adaptive penalty scaling based on the EI distribution
    # Scale by the 90th percentile of ei_cost to be robust to outliers
    ei_cost_p90 = np.percentile(ei_cost, 90)
    penalty_scale = max(ei_cost_p90, 1e-9) * 0.3
    
    # Final acquisition score
    score = ei_cost - penalty_scale * crowding_penalty
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
