"""Deterministic acquisition function: POI weighted by dynamic exploration, normalized by cost."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (POI)
    weighted by a dynamic exploration factor derived from global max uncertainty,
    normalized by cost to prioritize high-probability improvements in low-cost regions.
    
    Parameters:
    -----------
    mu : array_like, shape (N,)
        Mean predictions from GP in objective units.
    sigma : array_like, shape (N,)
        Standard deviation predictions from GP in objective units.
    f_best : float
        Best observed function value.
    cost : array_like, shape (N,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (N,)
        Acquisition scores (higher = more promising).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    eps = 1e-12
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Compute z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    poi = norm.cdf(z)
    
    # Dynamic exploration factor based on global maximum uncertainty
    # We want to balance exploitation (high POI) with exploration (high sigma)
    # Use a weight that increases with uncertainty but is normalized by the max sigma
    max_sigma = np.max(sigma_safe)
    if max_sigma < eps:
        max_sigma = eps
    
    # Exploration weight: scales with sigma relative to max uncertainty
    # This ensures we don't over-explore when uncertainty is low
    exploration_weight = sigma_safe / max_sigma
    
    # Combine POI and exploration
    # We use a soft combination: POI dominates when sigma is small,
    # exploration matters more when sigma is large
    # The weight alpha adapts based on the overall uncertainty level
    # When max_sigma is high relative to the scale of improvement, we explore more
    # Use a simple heuristic: alpha = 0.5 (balanced)
    # But make it adaptive: if the best improvement is large, exploit more
    improvement_scale = np.max(f_best - mu) if np.max(f_best - mu) > 0 else eps
    # Normalize improvement by sigma scale
    alpha = 0.5  # Fixed balance between exploitation and exploration
    
    # Combined score: weighted sum of POI and exploration
    combined = alpha * poi + (1 - alpha) * exploration_weight
    
    # Normalize by cost to prioritize low-cost evaluations
    # Add a small epsilon to avoid division by zero
    
    scores = combined / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
