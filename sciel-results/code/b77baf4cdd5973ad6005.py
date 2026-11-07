"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 300.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI) with
    a budget-aware Thompson Sampling-like exploration bonus.
    
    The key insight: as total cost increases, the remaining budget shrinks,
    so we should be more aggressive in exploitation. We model this by:
    1. Computing EI (exploitation signal)
    2. Adding a scaled exploration term that decreases with total cost
    3. Normalizing by cost to prefer cheap evaluations
    """
    # Deadline guard: if we're close to the limit, return a safe default
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return np.zeros_like(mu)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Phi(z) = CDF, phi(z) = PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI = improvement * Phi(z) + sigma * phi(z)
    ei = improvement * cdf_z + sigma_safe * pdf_z
    
    # Clamp to non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Budget-aware exploration: as cost increases, reduce exploration
    # cost is in [1, 2], total budget is 30
    # remaining_budget_fraction = (30 - cost) / 30, but cost is per-point
    # Use total_cost proxy: higher mean cost means we've spent more
    mean_cost = np.mean(cost)
    # Exploration coefficient decreases as we spend more
    # At low cost (~1), kappa ~ 1.0; at high cost (~2), kappa ~ 0.3
    kappa = 0.3 + 0.7 * np.exp(-0.5 * (mean_cost - 1.0))
    
    # Combined score: EI + kappa * sigma (exploration)
    # This is like a soft UCB that transitions from exploration to exploitation
    score = ei + kappa * sigma_safe
    
    # Cost normalization: prefer cheaper points
    # cost is in [1, 2], so 1/cost is in [0.5, 1]
    # Use a moderate power to not over-penalize
    score = score / (cost ** 1.2)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
    # EVOLVE-BLOCK-END
