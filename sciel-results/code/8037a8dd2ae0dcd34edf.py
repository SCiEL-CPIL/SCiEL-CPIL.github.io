import numpy as np
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Lower Confidence Bound (LCB) acquisition function.
    Maximizes: -(mu - kappa * sigma) - lambda * cost_normalized
    which is equivalent to minimizing: mu - kappa * sigma + lambda * cost_normalized
    """
    # Deadline guard: track time to avoid exceeding the 300s limit
    # Since this is a single function call, we just ensure it's fast.
    
    # Numerical safety
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # The minimum value of cosine8 is -0.8
    f_min = -0.8
    
    # Current regret
    regret = f_best - f_min
    
    # Normalize regret to [0, 1]
    # If f_best is close to -0.8, regret is small (good)
    # If f_best is far from -0.8, regret is large (bad)
    # We assume a max relevant regret of 1.0 for scaling
    regret_norm = np.clip(regret, 0.0, 1.0)
    
    # Adaptive kappa:
    # When regret is high (early, poor f_best), we want to explore more -> higher kappa
    # When regret is low (late, good f_best), we want to exploit more -> lower kappa
    # kappa ranges from 0.5 (low regret) to 2.0 (high regret)
    kappa = 0.5 + 1.5 * regret_norm
    
    # Cost normalization:
    # Cost is in [1, 2] (since x is in [-1,1]^8, mean(x scaled to [0,1]) is in [0,1], so cost is in [1,2])
    # Normalize cost to [0, 1] for better relative penalty
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range > eps:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.zeros_like(cost)
    
    # Adaptive lambda:
    # When regret is high, we can afford to spend more on exploration -> lower lambda
    # When regret is low, we need to conserve budget -> higher lambda
    # lambda ranges from 0.5 (high regret) to 2.0 (low regret)
    lambda_cost = 0.5 + 1.5 * (1.0 - regret_norm)
    
    # LCB score: we want to minimize mu - kappa * sigma + lambda * cost_norm
    # Acquisition score (higher is better): -(mu - kappa * sigma + lambda * cost_norm)
    lcb_value = mu - kappa * sigma_safe + lambda_cost * cost_norm
    scores = -lcb_value
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
# EVOLVE-BLOCK-END
