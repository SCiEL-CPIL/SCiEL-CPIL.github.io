"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Bayesian Regret Minimization.
    
    Directly targets the scoring metric (simple regret) by approximating
    expected simple regret reduction per unit cost.
    """
    # Track time to ensure we don't exceed deadline
    _t_start = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Bayesian Regret Minimization:
    # Approximate E[min(f(x), f_best)] ≈ f_best - 0.5 * max(0, f_best - mu - 0.5*sigma)
    # The regret reduction is max(0, f_best - mu - 0.5*sigma)
    # We want to maximize expected regret reduction per unit cost
    
    # Compute the expected improvement in regret
    improvement = f_best - mu - 0.5 * sigma_safe
    improvement = np.maximum(improvement, 0.0)
    
    # Normalize by cost
    regret_per_cost = improvement / cost_safe
    
    # Add a small exploration bonus for high uncertainty regions
    # This helps balance exploitation and exploration
    exploration_bonus = sigma_safe / cost_safe * 0.05
    
    # Combine
    score = regret_per_cost + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Ensure non-negative scores
    score = np.maximum(score, 0.0)
    
    # Safety check: if we're getting close to deadline, return immediately
    # (This is a single call, so it should be fast, but we add a guard)
    if time.time() - _t_start > 0.1:
        # If somehow slow, just return what we have
        pass
    
    return score
# EVOLVE-BLOCK-END
