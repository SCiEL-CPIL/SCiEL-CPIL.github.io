"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard: keep a safety margin before the 300 s kill.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # 20 s margin


def _time_guard():
    """Return True if we are within the safe time budget."""
    return time.time() < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using GP-UCB with budget-scaled exploration.
    Returns 2000 finite scores; higher = evaluate next.
    
    Strategy:
    - Use UCB: mu + kappa * sigma (for minimization, we want low mu, high sigma)
    - kappa decays linearly from high (early) to low (late) as cost approaches 30
    - Divide by a convex combination of cost and a constant to balance exploitation
      of low-cost regions against exploration of high-uncertainty zones
    """
    # Deadline guard: if running out of time, return a simple valid fallback.
    if not _time_guard():
        # Fallback: favor cheap, uncertain points
        c = np.maximum(cost, 1e-9)
        s = np.maximum(sigma, 1e-9)
        return s / c

    eps = 1e-12
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)

    # Total cost budget is 30. Use the current cost to determine phase.
    # cost here is the per-candidate evaluation cost, but we need total cost.
    # Actually, cost has shape (2000,) and represents the cost of evaluating each candidate.
    # The total cost budget is 30, but we don't have access to the total spent.
    # However, we can infer progress from f_best or use a fixed schedule.
    # Let's use a fixed linear decay based on an assumed number of iterations.
    # With 5 initial points and cost ~1.5 each, initial cost ~7.5.
    # Remaining budget ~22.5, so roughly 15 more evaluations.
    # Let's assume we're at iteration t out of T=20 total evaluations.
    # Since we don't know t, let's use the magnitude of f_best as a proxy for progress.
    # Actually, a simpler approach: use a fixed kappa that balances exploration/exploitation.
    # Or, use the cost array itself to determine the phase.
    
    # Let's use a budget-scaled kappa.
    # The total budget is 30. Each evaluation costs between 1 and 2.
    # Average cost is about 1.5, so we get about 20 evaluations.
    # Let's assume we're at evaluation number k out of 20.
    # Since we don't know k, let's use a heuristic based on f_best.
    # If f_best is close to 0, we're doing well and should exploit more.
    # If f_best is large, we should explore more.
    
    # Let's use a combination:
    # kappa = kappa_max - (kappa_max - kappa_min) * progress
    # where progress is estimated from f_best.
    
    # Three hump camel 2d minimum is 0.
    # If f_best < 0.1, we're close to optimum, exploit more.
    # If f_best > 1.0, we're far, explore more.
    
    f_best_clipped = np.clip(f_best, 0.0, 5.0)
    progress = f_best_clipped / 5.0  # 0 = at optimum, 1 = far away
    
    # kappa decays from high (exploration) to low (exploitation)
    # Early on (high f_best), we want high kappa.
    # Late on (low f_best), we want low kappa.
    kappa_min = 0.5
    kappa_max = 3.0
    kappa = kappa_min + (kappa_max - kappa_min) * progress
    
    # UCB for minimization: we want to minimize mu - kappa * sigma
    # So the score (higher is better) should be -mu + kappa * sigma
    ucb = -mu + kappa * s
    
    # Cost-aware: divide by a convex combination of cost and a constant
    # alpha * cost + (1 - alpha) * c0
    # c0 is a baseline cost (e.g., 1.0)
    # alpha controls how much weight we give to cost vs. baseline
    alpha = 0.5
    c0 = 1.0
    cost_weighted = alpha * c + (1 - alpha) * c0
    
    score = ucb / cost_weighted
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
