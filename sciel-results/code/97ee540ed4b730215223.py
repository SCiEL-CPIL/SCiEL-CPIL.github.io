"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 315.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: favor low cost and high uncertainty to finish gracefully
        return -cost + 0.1 * sigma

    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)

    # Calculate Z-score for Expected Improvement
    # Z = (f_best - mu) / sigma
    # If mu < f_best, Z is positive (potential improvement)
    Z = (f_best - mu) / sigma_safe
    
    # Clip Z to avoid numerical issues in CDF/PDF for extreme values
    # norm.cdf and norm.pdf are stable, but clipping prevents overflow in intermediate steps if any
    Z_clipped = np.clip(Z, -10, 10)
    
    Phi = norm.cdf(Z_clipped)
    phi = norm.pdf(Z_clipped)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    EI = (f_best - mu) * Phi + sigma_safe * phi
    
    # Exploration bonus
    # Add a term proportional to uncertainty to encourage exploration in 8D
    # This acts like a UCB term added to EI
    beta_explore = 1.5
    Bonus = beta_explore * sigma_safe
    
    # Total Value: Combines exploitation (EI) and exploration (Bonus)
    TotalValue = EI + Bonus
    
    # Cost-Aware Scoring (Stochastic Knapsack approach)
    # Maximize value per unit cost
    # This penalizes expensive evaluations (higher cost) if their value is not sufficiently higher
    scores = TotalValue / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
