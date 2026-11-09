"""Cost-aware acquisition function for Bayesian optimization on Rastrigin 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track start time for deadline guard
_t0 = time.time()
_DEADLINE = 325.0  # seconds, safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: blends expected improvement (exploitation)
    with uncertainty-driven exploration, weighted by evaluation cost.
    
    Parameters:
        mu: (2000,) GP mean in objective units
        sigma: (2000,) GP std in objective units
        f_best: scalar, best observed value
        cost: (2000,) evaluation cost = 1 + mean(x_scaled)
    
    Returns:
        (2000,) scores, higher = evaluate next
    """
    # Deadline guard
    elapsed = time.time() - _t0
    if elapsed > _DEADLINE:
        return np.zeros(2000)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Expected Improvement (exploitation)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Exploration term: use sigma directly, scaled
    # Rastrigin has local minima; high sigma means we haven't explored enough
    exploration = sigma
    
    # Budget-aware blend: estimate how much budget is left
    # cost is per-evaluation; total budget is 30, start with 5 points
    # We don't know exact budget used, but f_best and the loop structure tell us
    # Use a simple heuristic: if f_best is still high, we're early -> explore more
    # f_best for rastrigin can be large initially; once close to 0, exploit
    # Normalize: if f_best > 10, very early; if f_best < 1, late
    if f_best > 10.0:
        explore_weight = 0.6
    elif f_best > 5.0:
        explore_weight = 0.4
    elif f_best > 1.0:
        explore_weight = 0.2
    else:
        explore_weight = 0.1
    
    # Blend: weight EI and exploration
    # Scale exploration by a factor to make it comparable to EI
    # EI is in objective units; sigma is in objective units
    # Use a temperature-like scaling
    score = (1.0 - explore_weight) * ei + explore_weight * exploration
    
    # Cost-aware: divide by cost to penalize expensive evaluations
    score = score / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
