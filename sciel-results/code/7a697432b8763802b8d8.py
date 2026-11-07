"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_module_start_time = time.time()
_TIMEOUT_LIMIT = 300.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _module_start_time > _TIMEOUT_LIMIT:
        # Fallback: simple cost-aware mean
        return -mu / np.maximum(cost, 1e-9)

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Dynamic beta for exploration-exploitation balance
    # Higher beta when uncertainty (sigma) is high relative to the range of predictions (mu)
    mu_range = np.ptp(mu) + 1e-9
    mean_sigma = np.mean(sigma)
    
    # Scale beta based on relative uncertainty
    # If mean_sigma is large compared to mu_range, increase exploration
    beta = 1.0 + 2.0 * (mean_sigma / mu_range)
    # Clip beta to a reasonable range [1.0, 3.0]
    beta = np.clip(beta, 1.0, 3.0)
    
    # Lower Confidence Bound (LCB) for minimization
    # We want to minimize mu - beta * sigma
    # So we maximize score = beta * sigma - mu
    lcb_score = beta * sigma - mu
    
    # Cost-aware normalization
    # Using power 0.8 to soften the cost penalty compared to linear (1.0)
    # This allows slightly more expensive points if they offer high potential improvement
    cost_factor = np.power(cost, 0.8)
    
    # Final score
    scores = lcb_score / cost_factor
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
