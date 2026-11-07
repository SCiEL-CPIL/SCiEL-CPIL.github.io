"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware expected improvement with softened cost penalty.
    
    Parameters:
        mu: (2000,) GP mean predictions
        sigma: (2000,) GP standard deviation predictions
        f_best: float, best observed value so far
        cost: (2000,) evaluation cost for each candidate
    
    Returns:
        (2000,) finite scores, higher = evaluate next
    """
    # Deadline guard: if we're close to the time limit, return quickly
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Fallback: simple sigma-based exploration weighted by inverse cost
        # Use a softer cost penalty here too to maintain consistency
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-6)
        # Using alpha=0.5 for consistency with main logic
        scores = sigma_safe / (cost_safe ** 0.5)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Expected Improvement
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Clip z to avoid numerical issues with very large/small values
    # norm.cdf(-10) ~ 0, norm.cdf(10) ~ 1
    z_clipped = np.clip(z, -10, 10)
    
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    
    ei = diff * cdf + sigma_safe * pdf
    
    # EI should be non-negative (numerical guard)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score with softened cost exponent
    # The cost varies between 1 and 2. The global minimum is in the inner 80% of the domain.
    # A linear penalty (alpha=1) biases the search towards the cheaper half of the domain.
    # Since the minimum is uniformly distributed in the inner box, this bias is suboptimal.
    # Using alpha=0.5 reduces the bias, allowing exploration of the more expensive region.
    alpha = 0.5
    cost_penalty = cost_safe ** alpha
    
    score = ei / cost_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
