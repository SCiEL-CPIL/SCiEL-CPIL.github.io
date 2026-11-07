"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline management
_START_TIME = time.time()
_DEADLINE_SECONDS = 285.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        return np.zeros_like(mu, dtype=np.float64)

    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Handle any NaN or Inf values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)

    # Ensure sigma is strictly positive for division
    sigma = np.maximum(sigma, 1e-12)

    # Ensure cost is within reasonable bounds for Rastrigin task
    # Cost is 1 + mean(x_scaled), so [1, 2]
    cost = np.clip(cost, 1.0, 2.0)

    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)

    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)

    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)

    # Cost-aware EI: use a power penalty to amplify cost differences
    # cost in [1, 2], so cost^2 in [1, 4] gives up to 4x range
    # This makes cheap evaluations more attractive
    cost_penalty = cost ** 2.0
    cost_aware_ei = ei / cost_penalty

    # Add exploration bonus: prefer points with higher uncertainty
    # when EI is similar, explore more uncertain regions
    # Use a soft bonus that doesn't dominate EI
    exploration_bonus = sigma * 0.05

    # Combine: primarily cost-aware EI, with small exploration bonus
    score = cost_aware_ei + exploration_bonus

    # Normalize to [0, 1] range for stable scoring
    score_max = np.max(score)
    if score_max > 1e-12:
        normalized = score / score_max
    else:
        normalized = np.zeros_like(score)

    # Ensure finite output
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)

    return normalized
# EVOLVE-BLOCK-END
