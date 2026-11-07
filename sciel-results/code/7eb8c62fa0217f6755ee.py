# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining probability of improvement
    with a dynamic LCB exploration term that shifts from exploration to
    exploitation as the cost budget depletes.
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Probability of improvement (PI)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Dynamic kappa for LCB: starts with exploration, shifts to exploitation
    # cost is in [1, 2] range (1 + mean of scaled coords)
    # Total budget is 30, initial cost is ~5*1.5 = 7.5
    # So remaining budget fraction: (30 - cost) / 30
    # But we don't know total cost directly, only per-candidate cost
    # Use cost as a proxy: lower cost candidates are "cheaper"
    # We want to explore more when we have more budget left
    
    # Estimate remaining budget: assume we're at some point in the search
    # cost here is the cost of evaluating this specific candidate
    # Average cost is around 1.5, so ~20 evaluations possible after initial 5
    
    # Use a simple heuristic: kappa decreases as cost increases (more expensive = more exploitation)
    # cost in [1, 2], normalize to [0, 1]
    cost_norm = (cost - 1.0) / 1.0  # [0, 1]
    
    # kappa starts high (exploration) and drops to low (exploitation)
    # Use a smooth transition
    kappa = 2.0 * (1.0 - cost_norm) + 0.1
    
    # LCB: mu - kappa * sigma
    lcb = mu - kappa * sigma_safe
    
    # Combine PI and LCB:
    # PI gives confidence in improvement
    # LCB gives exploration bonus
    # Weight them: when mu is well below f_best, PI is high, prioritize exploitation
    # When uncertainty is high, LCB dominates
    
    # Score = PI * (f_best - mu) / cost + exploration_term / cost
    # Better: use a weighted combination
    improvement = f_best - mu
    improvement = np.maximum(improvement, 0)
    
    # PI-weighted improvement per unit cost
    score_exploit = pi * improvement / cost
    
    # Exploration per unit cost (higher when sigma is large and cost is low)
    score_explore = (kappa * sigma_safe) / cost
    
    # Combine: primarily exploitation, with exploration bonus
    score = score_exploit + 0.5 * score_explore
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
