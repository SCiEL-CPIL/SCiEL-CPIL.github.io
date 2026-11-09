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
    
    Combines Expected Improvement (EI) with a Probability of Improvement (PI)
    smoothing term and adaptive UCB exploration, all normalized by cost.
    The PI term provides a smoother exploration signal in multimodal regions,
    while EI drives exploitation. Adaptive weighting shifts from exploration
    (UCB + PI) early to exploitation (EI) as f_best approaches 0.
    
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
    
    # Expected Improvement (for minimization)
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (for minimization)
    # PI provides smoother gradient than EI for exploration
    pi = norm.cdf(z)
    
    # UCB (lower is better for minimization)
    # Adaptive beta: explore more when f_best is far from 0
    f_scale = 30.0
    f_norm = min(max(f_best / f_scale, 0.0), 1.0)
    
    # Beta: high early (exploration), low late (exploitation)
    beta = 0.5 + 2.0 * f_norm
    ucb = mu + beta * sigma
    
    # Cost: divide by cost to get per-unit-cost value
    # cost is in [1, 2] since 1 + mean(x_scaled) where x_scaled in [0,1]
    cost_safe = np.maximum(cost, _EPS)
    
    # Normalize each component to [0, 1] for fair blending
    ei_max = np.max(ei) if np.max(ei) > _EPS else 1.0
    ei_norm = ei / ei_max
    
    pi_max = np.max(pi) if np.max(pi) > _EPS else 1.0
    pi_norm = pi / pi_max
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min if ucb_max - ucb_min > _EPS else 1.0
    ucb_norm = (ucb_max - ucb) / ucb_range  # [0, 1], higher = lower UCB = better
    
    # Adaptive weighting:
    # Early on (high f_norm): weight UCB and PI more for exploration
    # Late (low f_norm): weight EI more for exploitation
    # w_ei: 0.25 (early) -> 0.75 (late)
    # w_pi: 0.25 (early) -> 0.15 (late) - PI is auxiliary
    # w_ucb: 0.5 (early) -> 0.1 (late)
    w_ei = 0.25 + 0.50 * (1.0 - f_norm)
    w_pi = 0.25 - 0.10 * (1.0 - f_norm)
    w_ucb = 1.0 - w_ei - w_pi
    
    # Ensure weights are non-negative
    w_ei = max(w_ei, 0.0)
    w_pi = max(w_pi, 0.0)
    w_ucb = max(w_ucb, 0.0)
    
    # Cost-adjusted scoring:
    # Use sqrt(cost) for PI and UCB (softer cost penalty for exploration terms)
    # Use cost for EI (stronger cost penalty for exploitation)
    cost_sqrt = np.sqrt(cost_safe)
    
    score = (w_ei * ei_norm / cost_safe + 
             w_pi * pi_norm / cost_sqrt + 
             w_ucb * ucb_norm / cost_sqrt)
    
    # Add a small bonus for very low cost to break ties
    cost_bonus = 0.05 * (1.0 / cost_safe - 0.5)
    score += cost_bonus
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
