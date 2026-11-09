"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: 330s wall-clock limit, return safely before it.
    t0 = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero or near-zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-normalized probability of improvement (POI) combined with
    # a deterministic regret lower bound: (f_best - mu + 0.5*sigma) / cost
    # This directly optimizes expected reduction in simple regret per unit cost.
    
    # POI for minimization: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Deterministic regret lower bound: f_best - mu + 0.5 * sigma
    # This is a conservative lower bound on expected improvement that
    # accounts for uncertainty (0.5 sigma ≈ 50th percentile of improvement)
    regret_lower = f_best - mu + 0.5 * sigma_safe
    
    # Combine: weight POI by the regret lower bound, normalized by cost
    # Higher scores = better candidates to evaluate next
    scores = poi * regret_lower / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time: if we've used more than 325s, return immediately
    if time.time() - t0 > 325:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
