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
    
    Combines Expected Improvement (EI), Probability of Improvement (PI), and UCB
    exploration with adaptive cost-aware weighting. EI drives exploitation, PI
    provides smooth exploration signal, and UCB handles multimodal regions.
    Cost^0.5 normalization creates a balanced trade-off between finding good
    values and staying within budget.
    
    mu, sigma: (2000,) arrays of GP mean and std in objective units
    f_best: scalar, lowest observed value
    cost: (2000,) array of evaluation costs
    
    Returns: (2000,) finite scores, higher = prefer to evaluate next
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 300s limit
    
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
    
    # Probability of Improvement (for minimization): P(mu < f_best)
    pi = norm.cdf(z)
    
    # UCB (lower is better for minimization)
    # Adaptive beta: explore more when f_best is far from 0
    # Ackley 2D: f_best typically starts around 10-20, minimum is 0
    f_scale = 30.0
    f_norm = min(max(f_best / f_scale, 0.0), 1.0)
    
    # Beta: high early (exploration), low late (exploitation)
    # Smooth transition using sigmoid-like curve
    beta = 0.3 + 2.5 * (f_norm ** 0.7)
    
    ucb = mu + beta * sigma
    
    # Cost-adjusted scoring
    cost_safe = np.maximum(cost, _EPS)
    
    # Normalize each component to [0, 1]
    ei_max = np.max(ei) if np.max(ei) > _EPS else 1.0
    ei_norm = ei / ei_max
    
    # PI is already in [0, 1]
    pi_norm = pi
    
    # UCB: lower is better. Normalize so that lower UCB gives higher score
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min if ucb_max - ucb_min > _EPS else 1.0
    ucb_norm = (ucb_max - ucb) / ucb_range
    
    # Adaptive weighting with smooth transition:
    # Early (high f_norm): emphasize UCB and PI for broad exploration
    # Late (low f_norm): emphasize EI for precise exploitation
    # Use smoothstep for better transition
    t = 1.0 - f_norm  # 0 early, 1 late
    t_smooth = t * t * (3.0 - 2.0 * t)  # smoothstep
    
    w_ei = 0.15 + 0.65 * t_smooth
    w_pi = 0.35 - 0.20 * t_smooth
    w_ucb = 0.50 - 0.45 * t_smooth
    
    # Ensure weights are non-negative and sum to ~1
    w_ei = max(w_ei, 0.05)
    w_pi = max(w_pi, 0.05)
    w_ucb = max(w_ucb, 0.05)
    w_sum = w_ei + w_pi + w_ucb
    w_ei /= w_sum
    w_pi /= w_sum
    w_ucb /= w_sum
    
    # Score: blend of cost-adjusted components
    # Use cost^0.5 in denominator for smoother cost trade-off
    # This allows more exploration in early iterations when cost matters less
    cost_adj = np.sqrt(cost_safe)
    score = (w_ei * ei_norm + w_pi * pi_norm + w_ucb * ucb_norm) / cost_adj
    
    # Add a small exploration bonus for high-uncertainty regions
    # This helps escape local minima in multimodal landscapes
    sigma_norm = sigma / (np.max(sigma) + _EPS)
    exploration_bonus = 0.03 * sigma_norm * (1.0 - t_smooth)
    score += exploration_bonus
    
    # Add a small bonus for very low cost to break ties
    # cost is in [1, 2], so (1/cost - 0.5) is in [0, 0.5]
    cost_bonus = 0.05 * (1.0 / cost_safe - 0.5)
    score += cost_bonus
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
