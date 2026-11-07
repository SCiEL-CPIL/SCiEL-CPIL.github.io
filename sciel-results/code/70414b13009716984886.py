# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a multiplicative feasibility gate.
    
    The score is Expected Improvement (EI) modulated by a smooth decay factor
    based on the candidate's cost relative to the median cost of the candidate
    pool. This encourages exploitation of low-cost regions while still allowing
    exploration of higher-cost regions (like the center where the optimum lies)
    if the Expected Improvement is sufficiently high.
    """
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-awareness: Use a multiplicative gate based on cost.
    # We use the median (50th percentile) of the candidate costs as the baseline.
    # This allows points around the average cost (which includes the optimum)
    # to pass through with minimal penalty, while strongly penalizing the most
    # expensive points (top right corner).
    cost_threshold = np.percentile(cost, 50) + 1e-9
    
    # Ratio of candidate cost to the median cost
    ratio = cost / cost_threshold
    
    # Multiplicative decay: 
    # For ratio <= 1.0 (cheaper than median), factor is 1.0.
    # For ratio > 1.0 (more expensive), factor decays exponentially.
    # We use a moderate decay rate (5.0) to allow some exploration of 
    # higher-cost areas if EI is high, but strongly discourage the most expensive.
    decay = np.where(
        ratio <= 1.0,
        1.0,
        np.exp(-5.0 * (ratio - 1.0) ** 2)
    )
    
    scores = ei * decay
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
