"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_EPS = 1e-12

def _ackley_2d(x):
    """Ackley 2D function (standard, minimum at 0,0)."""
    x1 = x[:, 0]
    x2 = x[:, 1]
    sum_sq = x1**2 + x2**2
    sum_cos = np.cos(2.0 * np.pi * x1) + np.cos(2.0 * np.pi * x2)
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * sum_sq)) - np.exp(0.5 * sum_cos) + np.e + 20.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses cost-aware Expected Improvement (EI/cost) as the primary signal,
    combined with a small uncertainty-based exploration bonus. This creates
    a strong preference for high-improvement, low-cost points while still
    allowing exploration of uncertain regions.
    
    mu, sigma: (2000,) arrays of GP mean and std in objective units
    f_best: scalar, lowest observed value
    cost: (2000,) array of evaluation costs
    
    Returns: (2000,) finite scores, higher = prefer to evaluate next
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, _EPS)
    cost_safe = np.maximum(cost, _EPS)
    
    # Expected Improvement (for minimization)
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI: primary signal
    cai = ei / cost_safe
    
    # Exploration bonus: reward points with high uncertainty that are
    # reasonably close to the best (not too far off). This helps escape
    # local optima in multimodal Ackley landscape.
    # Use a soft cap to avoid over-exploring far-away uncertain regions.
    # Scale: when f_best is large (early), explore more; when small (late), exploit more.
    f_scale = 20.0
    f_norm = min(max(f_best / f_scale, 0.0), 1.0)
    
    # Exploration weight: higher early, lower late
    w_explore = 0.3 * f_norm + 0.05
    
    # Exploration bonus: sigma-based, but discounted for points that are
    # unlikely to improve (mu >> f_best)
    # Use z-score: points with z < 0 (mu < f_best) are promising
    # Bonus is highest for moderate uncertainty in promising regions
    explore_bonus = w_explore * sigma * norm.pdf(z) / cost_safe
    
    # Combine: cost-aware EI + exploration bonus
    score = cai + explore_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
