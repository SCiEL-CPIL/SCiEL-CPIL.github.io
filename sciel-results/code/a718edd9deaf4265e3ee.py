"""Cost-aware acquisition function for Bayesian optimization (Shekel 4D)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: cost-normalized Lower Confidence Bound (LCB).
    
    score = -(mu - kappa * sigma) / cost
    
    where kappa dynamically balances exploitation vs exploration based on
    the spread of the predictive distribution.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        cost_safe = np.maximum(cost, 1e-8)
        fallback = -mu / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Safe arrays
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Dynamic kappa: scale exploration with the relative uncertainty
    # Use a fixed kappa=1.0 as suggested, but normalize by the typical sigma
    # so that the exploration bonus is meaningful in objective units.
    # Actually, the directive says kappa=1.0 directly. Let's use that.
    kappa = 1.0
    
    # LCB: lower confidence bound on the function value
    lcb = mu - kappa * sigma_safe
    
    # Cost-normalized score: higher is better
    scores = -lcb / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
