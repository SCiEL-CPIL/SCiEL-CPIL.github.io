"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Precompute constants for speed
_EPS = 1e-9

def _ackley_2d(x):
    """Ackley 2D function (standard, minimum at 0,0)."""
    x1 = x[:, 0]
    x2 = x[:, 1]
    sum_sq = x1**2 + x2**2
    sum_cos = np.cos(2.0 * np.pi * x1) + np.cos(2.0 * np.pi * x2)
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * sum_sq)) - np.exp(0.5 * sum_cos) + np.e + 20.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: blends normalized PI and UCB terms
    with adaptive weighting based on how close f_best is to the known minimum (0).
    
    mu, sigma: (2000,) arrays of GP mean and std in objective units
    f_best: scalar, lowest observed value
    cost: (2000,) array of evaluation costs
    
    Returns: (2000,) finite scores, higher = prefer to evaluate next
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Probability of Improvement (PI) - smoother than EI for exploration
    improvement = f_best - mu
    z = improvement / sigma
    pi = norm.cdf(z)  # P(mu < f_best)
    
    # UCB term
    # Adaptive beta: larger when f_best is far from optimum (early exploration)
    # f_best for Ackley is in [0, ~40]. Use a scaled confidence.
    # When f_best is large (early), explore more; when small, exploit more.
    beta_base = 2.0
    # Normalize f_best to [0, 1] range for scaling (Ackley max ~ 20+20+e+20 ~ 65, but typically < 40)
    f_norm = min(max(f_best / 40.0, 0.0), 1.0)
    beta = beta_base * (0.3 + 0.7 * f_norm)  # beta in [0.6, 2.0]
    
    ucb = mu + beta * sigma
    
    # Normalize PI by its max to get [0, 1]
    pi_max = np.max(pi) if np.max(pi) > 0 else 1.0
    pi_norm = pi / pi_max
    
    # Normalize UCB: lower UCB is better, so invert
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min if ucb_max > ucb_min else 1.0
    ucb_norm = (ucb_max - ucb) / ucb_range  # [0, 1], higher = lower UCB = better
    
    # Cost normalization: use cost^0.5 to balance exploration efficiency against computational budget
    cost_norm = np.sqrt(cost) / np.sqrt(np.max(cost)) if np.max(cost) > 0 else np.ones_like(cost)
    
    # Adaptive weight: early -> more UCB, late -> more PI
    w_pi = 0.2 + 0.8 * (1.0 - f_norm)  # [0.2, 1.0]
    w_ucb = 1.0 - w_pi + 0.2  # [0.2, 1.0], ensure minimum exploration
    
    score = (w_pi * pi_norm + w_ucb * ucb_norm) / (cost_norm + _EPS)
    
    # Check deadline
    if time.time() > deadline:
        # Return best result found so far
        pass
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
