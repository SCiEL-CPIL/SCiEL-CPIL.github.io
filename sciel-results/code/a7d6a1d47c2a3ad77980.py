"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Deterministic Worst-Case Improvement (WCI) per unit cost with adaptive
    exploration via quantile-scaled sigma penalty.
    
    WCI = max(0, f_best - mu - k * sigma)
    Score = WCI / cost
    
    The exploration constant k is adapted based on the current spread of
    predicted uncertainties to balance exploitation and exploration.
    """
    # Adaptive exploration constant: scale k based on the relative magnitude
    # of sigma compared to the improvement potential
    sigma_max = np.max(sigma)
    sigma_median = np.median(sigma)
    
    # If sigma is very large relative to the gap to f_best, increase exploration
    # If sigma is small, decrease exploration to exploit
    gap = f_best - np.min(mu)
    if gap > 1e-12:
        # Normalize sigma by the gap to get a relative uncertainty measure
        rel_unc = sigma_median / gap
        # Map relative uncertainty to k in [0.5, 2.5]
        k = 0.5 + 2.0 * np.clip(rel_unc, 0.0, 1.0)
    else:
        k = 1.0
    
    # Compute worst-case improvement (deterministic, no CDF/PDF)
    wci = np.maximum(0.0, f_best - mu - k * sigma)
    
    # Guard against zero or negative costs
    cost_safe = np.maximum(cost, 1e-9)
    
    # Score: improvement per unit cost (higher = evaluate next)
    scores = wci / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
