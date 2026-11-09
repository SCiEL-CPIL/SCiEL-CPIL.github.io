"""Cost-aware acquisition: diversity-aware scoring that penalizes spatial
redundancy by leveraging the GP's predictive structure. Points with high
sigma are in uncertain regions; we boost them but also apply a soft penalty
for candidates that are likely correlated with already-uncertain regions.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def _elapsed_ok():
    return time.time() < _DEADLINE - 3.0

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    n = len(mu)
    cost_safe = np.maximum(cost, 1e-9)

    # Robust sigma floor
    mu_min = float(np.min(mu))
    span = max(f_best - mu_min, 1e-9)
    sigma_floor = max(1e-6, 0.03 * span)
    sigma_safe = np.maximum(sigma, sigma_floor)

    if not _elapsed_ok():
        score = (f_best - mu) / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return score

    # Core improvement signal: expected improvement potential
    improvement = f_best - mu
    improvement = np.maximum(improvement, 0.0)

    # Uncertainty bonus: reward high-sigma regions
    # Normalize sigma by its own scale for stability
    sigma_mean = float(np.mean(sigma_safe))
    sigma_scale = max(sigma_mean, 1e-9)
    sigma_norm = sigma_safe / sigma_scale

    # Diversity-aware penalty:
    # We approximate spatial redundancy by noting that in a GP with length scale
    # ~0.2 on [0,1]^d, points close to each other have highly correlated predictions.
    # We can't compute pairwise distances (no coordinates), but we CAN use the
    # structure of sigma: if many points have similar high sigma, they likely
    # occupy the same uncertain region. We down-weight points whose sigma is
    # "typical" of the high-sigma cluster, and boost those that are outliers
    # in sigma space (potentially in different uncertain regions).
    
    # Compute how "unique" each point's uncertainty level is
    # Points with sigma far from the median are in distinct regions
    sigma_median = float(np.median(sigma_safe))
    sigma_spread = float(np.std(sigma_safe)) + 1e-9
    
    # Z-score of sigma: points with unusual sigma get a diversity bonus
    sigma_z = (sigma_safe - sigma_median) / sigma_spread
    
    # Diversity bonus: positive for high-sigma outliers (new uncertain regions),
    # slightly negative for low-sigma outliers (explored regions, less interesting)
    diversity_bonus = np.where(sigma_z > 0, 0.15 * sigma_z, 0.05 * sigma_z)

    # Combined score:
    # Primary: improvement / cost (exploitation efficiency)
    # Secondary: uncertainty bonus scaled by cost
    # Tertiary: diversity bonus to encourage exploring distinct regions
    
    kappa = 1.8  # uncertainty weight
    alpha = 1.0  # cost penalty exponent
    
    exploitation = improvement / cost_safe
    exploration = kappa * sigma_norm / cost_safe
    diversity = diversity_bonus / cost_safe

    score = exploitation + exploration + diversity

    # Numerical safety
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)

    return score
# EVOLVE-BLOCK-END
