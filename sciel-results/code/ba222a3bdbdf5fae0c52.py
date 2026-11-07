"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining EI and UCB for robust optimization.
    Uses a soft cost penalty to allow exploration of moderately expensive regions.
    
    Parameters:
        mu: (2000,) GP mean predictions
        sigma: (2000,) GP standard deviation predictions
        f_best: float, best observed value so far
        cost: (2000,) evaluation cost for each candidate
    
    Returns:
        (2000,) finite scores, higher = evaluate next
    """
    # Deadline guard: if we're close to the time limit, return a fast fallback
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-6)
        # Fallback: pure exploration per cost
        fallback = sigma_safe / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Expected Improvement (EI)
    diff = f_best - mu
    z = diff / sigma_safe
    # Clip z to avoid numerical overflow/underflow in exp/cdf
    z_clipped = np.clip(z, -15, 15)
    
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    
    ei = diff * cdf + sigma_safe * pdf
    # EI is theoretically >= 0, but numerical errors can make it slightly negative
    ei = np.maximum(ei, 0.0)
    
    # Upper Confidence Bound (UCB)
    # Beta controls exploration. 1.5 is a reasonable default for 2D Ackley.
    beta = 1.5
    ucb = -mu + beta * sigma_safe
    
    # Blend EI and UCB
    # 70% exploitation (EI), 30% exploration (UCB)
    blended = 0.7 * ei + 0.3 * ucb
    
    # Soft Cost Penalty
    # The raw cost is in [1, 2] (since x is scaled to [0,1], mean(x) in [0,1], cost = 1 + mean(x)).
    # A strict 1/cost penalty might over-penalize expensive regions that contain the minimum.
    # Soft cost: 1 + 0.4 * (cost - 1). Range: [1, 1.4].
    # This reduces the penalty for higher costs, allowing the algorithm to consider
    # points with slightly higher cost if their acquisition value is high enough.
    cost_soft = 1.0 + 0.4 * np.clip(cost - 1.0, 0.0, 1.0)
    cost_soft = np.maximum(cost_soft, 1e-6)
    
    score = blended / cost_soft
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
