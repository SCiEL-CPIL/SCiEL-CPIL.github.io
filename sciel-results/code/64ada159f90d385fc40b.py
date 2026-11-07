# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a multiplicative feasibility gate.
    
    The score is Expected Improvement (EI) only if the predicted cost is below
    a dynamically determined threshold (median cost). This creates a budget-aware
    gate that balances exploration of the low-cost corner with the likelihood
    of the optimum being in the mid-cost region (inner 80% of the domain).
    """
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-awareness: use a multiplicative gate based on cost.
    # We use the median (50th percentile) of costs as the dynamic threshold.
    # This is more appropriate for Levy 2D with a moving minimum in the inner 80%
    # than the 25th percentile, which was too restrictive and biased towards
    # the extreme low-cost corner.
    cost_threshold = np.percentile(cost, 50) + 1e-9
    
    # Gate: only pass EI if cost is below threshold; otherwise penalize.
    # Use a smooth decay for points above the threshold to allow some
    # exploration of slightly more expensive regions if the GP is confident.
    ratio = cost / (cost_threshold + 1e-9)
    
    # Multiplicative decay: for ratio <= 1, factor = 1.0; for ratio > 1,
    # factor decays as exp(-2.0 * (ratio - 1)^2).
    # A softer decay (2.0 vs 10.0) allows the search to consider points
    # slightly above the median cost, which is crucial if the optimum
    # is located in the upper half of the cost distribution.
    decay = np.where(
        ratio <= 1.0,
        1.0,
        np.exp(-2.0 * (ratio - 1.0) ** 2)
    )
    
    scores = ei * decay
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
