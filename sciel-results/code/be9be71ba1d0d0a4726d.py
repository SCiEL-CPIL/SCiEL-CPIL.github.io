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
    Cost-aware acquisition function using Probability of Improvement (PI).
    
    PI provides a smoother, more stable gradient for exploration in multimodal
    landscapes compared to EI. We normalize PI by cost^0.5 to balance
    exploration efficiency against computational budget.
    
    mu, sigma: (2000,) arrays of GP mean and std in objective units
    f_best: scalar, lowest observed value
    cost: (2000,) array of evaluation costs
    
    Returns: (2000,) finite scores, higher = prefer to evaluate next
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 300s limit (killed at 330s)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, _EPS)
    
    # Probability of Improvement (for minimization)
    # P(f(x) < f_best) = P(mu - sigma*Z < f_best) = P(Z > (mu - f_best)/sigma)
    # where Z ~ N(0,1)
    improvement = f_best - mu
    z = improvement / sigma
    # PI = P(f(x) < f_best) = norm.cdf(z)
    pi = norm.cdf(z)
    
    # Cost-adjusted scoring:
    # Normalize PI by cost^0.5 to balance exploration efficiency against budget
    # cost is in [1, 2] since 1 + mean(x_scaled) where x_scaled in [0,1]
    cost_safe = np.maximum(cost, _EPS)
    cost_factor = np.sqrt(cost_safe)
    
    # Score: PI normalized by cost^0.5
    # Higher PI means more likely to improve, lower cost means cheaper evaluation
    score = pi / cost_factor
    
    # Add a small exploitation term: prefer points with lower predicted mean
    # This helps break ties and guides toward the actual minimum
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min if mu_max - mu_min > _EPS else 1.0
    mu_exploit = (mu_max - mu) / mu_range  # [0, 1], higher = lower mu = better
    
    # Adaptive weighting based on how close f_best is to 0
    # f_best ranges from ~0 (converged) to ~20+ (early)
    f_scale = 30.0
    f_norm = min(max(f_best / f_scale, 0.0), 1.0)
    
    # w_pi goes from 0.6 (early, more exploration) to 0.8 (late, more exploitation via PI)
    # w_mu goes from 0.4 (early) to 0.2 (late)
    w_pi = 0.6 + 0.2 * (1.0 - f_norm)
    w_mu = 1.0 - w_pi
    
    # Blend PI-based score and exploitation term
    # Normalize both to [0, 1] range for fair blending
    pi_max = np.max(pi) if np.max(pi) > _EPS else 1.0
    pi_norm = pi / pi_max
    
    # Final score
    score = w_pi * pi_norm / cost_factor + w_mu * mu_exploit
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
