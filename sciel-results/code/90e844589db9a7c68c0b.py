import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Discrete, cost-normalized quantile-based heuristic for Rosenbrock 2D.
    
    Ranks candidates by the ratio of posterior probability of improvement (POI)
    to cumulative cost, decoupling from noisy mean estimates and prioritizing
    low-cost regions with high relative improvement likelihood.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, the best observed value so far.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores. Higher is better.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    n = len(mu)
    
    # Compute Probability of Improvement (POI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Ensure POI is in [0, 1]
    poi = np.clip(poi, 0.0, 1.0)
    
    # Cost normalization: normalize cost to [0, 1] range
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range > 1e-10:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.ones_like(cost)
    
    # Add small epsilon to avoid division by zero
    cost_safe = cost_norm + 1e-6
    
    # Core score: POI / cost_ratio
    # This prioritizes low-cost regions with high probability of improvement
    score = poi / cost_safe
    
    # Apply a mild exploitation boost for candidates where mu is already below f_best
    # This helps navigate the Rosenbrock valley by favoring regions with low predicted values
    # Use a smooth sigmoid-like transition
    mu_diff = f_best - mu  # Positive when mu < f_best
    exploit_boost = np.tanh(mu_diff / (sigma_safe + 1e-10)) * 0.1
    score += exploit_boost
    
    # Convert to rank-based scores for robustness
    # Higher score is better, so rank in descending order
    ranks = np.argsort(np.argsort(-score))
    
    # Final scores: use rank as the score (higher rank = better)
    # This ensures finite values and stable selection
    final_scores = n - 1 - ranks
    
    # Ensure all values are finite
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    return final_scores
