import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function.
    
    Computes EI(x) / cost(x) for each candidate, where EI is the expected
    improvement over f_best under the GP posterior. A small exploration
    bonus based on sigma is added to encourage exploration in uncertain
    regions. Higher scores indicate better candidates to evaluate next.
    """
    # Explicit deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: simple cost-aware EI with safe numerics
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure safety in denominators
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Compute Expected Improvement
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # We want to minimize f, so improvement is f_best - f(x) when f(x) < f_best
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # EI should be non-negative; clip small negative values from numerical error
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    scores = ei / cost_safe
    
    # Add a small exploration bonus to prevent premature convergence
    # This encourages visiting high-uncertainty regions even if EI is low
    # Scale bonus relative to typical EI magnitude
    sigma_normalized = sigma_safe / (np.max(sigma_safe) + 1e-12)
    exploration_bonus = 0.1 * sigma_normalized / cost_safe
    
    scores = scores + exploration_bonus
    
    # Ensure all outputs are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
