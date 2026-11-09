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
    
    Uses Probability of Improvement (PI) as the core signal, which provides
    a smoother gradient in multimodal landscapes. Combines with UCB-style
    exploration and cost-aware normalization using sqrt(cost) for gentler
    penalty. Adaptive weighting shifts from exploration to exploitation
    as f_best approaches the optimum.
    
    mu, sigma: (2000,) arrays of GP mean and std in objective units
    f_best: scalar, lowest observed value
    cost: (2000,) array of evaluation costs
    
    Returns: (2000,) finite scores, higher = prefer to evaluate next
    """
    start_time = time.time()
    deadline = start_time + 245.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, _EPS)
    
    # Compute improvement over f_best (positive when mu < f_best)
    improvement = f_best - mu
    z = improvement / sigma
    
    # Core acquisition: Probability of Improvement (smoother than EI)
    pi = norm.cdf(z)
    
    # Expected Improvement as secondary signal
    ei = improvement * pi + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB-style exploration term: mu - beta * sigma (lower is better for minimization)
    # Adaptive beta: more exploration when f_best is far from 0
    f_scale = 20.0
    f_norm = min(max(f_best / f_scale, 0.0), 1.0)
    beta = 0.2 + 1.2 * f_norm
    lb = mu - beta * sigma
    
    # Normalize all components to [0, 1]
    pi_min, pi_max = np.min(pi), np.max(pi)
    pi_range = pi_max - pi_min if pi_max - pi_min > _EPS else 1.0
    pi_norm = (pi - pi_min) / pi_range
    
    ei_max = np.max(ei) if np.max(ei) > _EPS else 1.0
    ei_norm = ei / ei_max
    
    lb_min, lb_max = np.min(lb), np.max(lb)
    lb_range = lb_max - lb_min if lb_max - lb_min > _EPS else 1.0
    lb_norm = (lb_max - lb) / lb_range  # Invert: lower lb -> higher score
    
    # Cost normalization using sqrt for gentler penalty
    cost_safe = np.maximum(cost, _EPS)
    cost_sqrt = np.sqrt(cost_safe)
    cost_sqrt_max = np.max(cost_sqrt) if np.max(cost_sqrt) > _EPS else 1.0
    cost_norm = cost_sqrt / cost_sqrt_max
    
    # Adaptive weighting:
    # Early (high f_norm): emphasize exploration (lb_norm, pi exploration)
    # Late (low f_norm): emphasize exploitation (pi, ei)
    w_pi = 0.4 + 0.3 * (1.0 - f_norm)
    w_ei = 0.2 + 0.2 * (1.0 - f_norm)
    w_lb = 0.2 + 0.2 * f_norm
    w_cost = 0.15 + 0.05 * (1.0 - f_norm)
    
    # Combined score: higher is better
    score = w_pi * pi_norm + w_ei * ei_norm + w_lb * lb_norm - w_cost * cost_norm
    
    # Add small bonus for very low cost to break ties
    cost_bonus = 0.05 * (1.0 / cost_safe - 0.5)
    score += cost_bonus
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
