# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using an adaptive multiplicative feasibility gate.
    
    The score is Expected Improvement (EI) modulated by a gate that allows evaluation
    if the cost is comparable to the average cost of the most promising candidates (top EI).
    This dynamically adjusts the "affordable" cost threshold based on the GP's beliefs,
    allowing exploration of expensive regions if the potential improvement is high enough
    to make them the top candidates.
    """
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive Threshold:
    # Identify the top candidates by Expected Improvement.
    # The "affordability" threshold is set to the mean cost of these top candidates,
    # potentially allowing slightly higher costs to account for variance.
    
    # Use top 100 candidates to be robust to noise
    top_k = 100
    if len(ei) >= top_k:
        top_indices = np.argpartition(ei, -top_k)[-top_k:]
        top_costs = cost[top_indices]
        # Use mean of top costs as the baseline
        adaptive_threshold = np.mean(top_costs)
        # Add a small buffer (10%) to allow some flexibility
        adaptive_threshold *= 1.1
    else:
        # Fallback if not enough candidates (should not happen with 2000)
        adaptive_threshold = np.mean(cost) * 1.1
    
    # Ensure threshold is at least the minimum possible cost to avoid division issues
    adaptive_threshold = np.maximum(adaptive_threshold, 1.0)
    
    # Gate: Calculate ratio of candidate cost to adaptive threshold
    ratio = cost / (adaptive_threshold + 1e-9)
    
    # Multiplicative decay:
    # For ratio <= 1.0 (cheaper than top candidates' average), factor = 1.0.
    # For ratio > 1.0 (more expensive), factor decays rapidly.
    # This creates a soft but strong preference for points that are not significantly
    # more expensive than the current "best bet" region.
    decay = np.where(
        ratio <= 1.0,
        1.0,
        np.exp(-10.0 * (ratio - 1.0) ** 2)
    )
    
    scores = ei * decay
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
