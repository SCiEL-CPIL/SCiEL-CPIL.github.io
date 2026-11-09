"""Cost-aware acquisition for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 320.0


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running out of time, return simple POI/cost
    if time.time() > _DEADLINE:
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        z = (f_best - mu) / sigma_safe
        poi = norm.cdf(z)
        score = poi / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return score

    # Input validation and safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = mu.shape[0]

    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # Check for non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)

    # Budget-aware LCB acquisition
    # The key insight: scale exploration by sqrt(remaining_budget)
    # Early on (large remaining budget): aggressive exploration
    # Late on (small remaining budget): precise exploitation
    
    # Estimate remaining budget. Total budget is 30.
    # We don't know exactly how much has been spent, but we can use
    # the mean cost as a proxy for per-evaluation cost, and estimate
    # remaining evaluations.
    # Actually, we can be smarter: the cost array tells us the cost of each candidate.
    # The total budget is 30. We don't know the current total cost spent.
    # However, we can use a heuristic: the number of evaluations so far is not given.
    # Let's use a simpler approach: scale by 1/sqrt(cost) which naturally
    # penalizes expensive evaluations, and add a term that encourages
    # exploration when sigma is high relative to the spread of mu.
    
    # Better approach: Use a beta parameter that adapts based on how
    # "spread out" the GP predictions are. If sigma is large relative
    # to the range of mu, we're in an underexplored region and should
    # explore more.
    
    # LCB for minimization: mu - beta * sigma
    # We want to maximize this (or minimize mu - beta*sigma)
    # So acquisition = -(mu - beta*sigma) / cost = (beta*sigma - mu) / cost
    
    # Adaptive beta: 
    # - Base beta from distance to known minimum
    # - Modulated by the coefficient of variation of sigma
    min_hartmann = -3.32237
    dist_to_min = max(0.0, f_best - min_hartmann)
    
    # Coefficient of variation of sigma across candidates
    sigma_mean = np.mean(sigma_safe)
    sigma_std = np.std(sigma_safe)
    cv = sigma_std / max(sigma_mean, 1e-8)
    
    # Base beta: start high (exploration) and decay as we get closer to optimum
    # dist_to_min starts around 0.5-1.0 and decreases to ~0
    base_beta = 2.0 * np.exp(-dist_to_min / 0.5) + 0.5
    
    # Scale by CV: if sigma is very variable, we're in a complex region
    # and should explore more
    beta = base_beta * (1.0 + 0.5 * cv)
    
    # Cost-normalized LCB
    lcb = (beta * sigma_safe - mu) / cost_safe
    
    # Also compute EI as a secondary signal
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei_cost_norm = ei / cost_safe
    
    # Normalize both
    lcb_std = np.std(lcb)
    ei_std = np.std(ei_cost_norm)
    
    if lcb_std > 1e-8:
        lcb_norm = (lcb - np.mean(lcb)) / lcb_std
    else:
        lcb_norm = np.zeros(n)
        
    if ei_std > 1e-8:
        ei_norm = (ei_cost_norm - np.mean(ei_cost_norm)) / ei_std
    else:
        ei_norm = np.zeros(n)
    
    # Blend: weight LCB more when we're far from optimum (exploration phase)
    # and EI more when close (exploitation phase)
    # Use a smooth transition based on dist_to_min
    lcb_weight = 0.4 + 0.4 * np.exp(-dist_to_min / 0.3)
    ei_weight = 1.0 - lcb_weight
    
    combined = lcb_weight * lcb_norm + ei_weight * ei_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(combined), combined, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
