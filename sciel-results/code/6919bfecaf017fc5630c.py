"""Rewritten cost-aware acquisition for Rastrigin 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _safe_norm_cdf(z):
    """Numerically stable normal CDF."""
    return 0.5 * (1.0 + np.sign(z) * np.sqrt(1.0 - np.exp(-2.0 * z * z + np.log(2.0))))


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining EI (exploitation) and UCB (exploration)
    with adaptive weighting based on how close f_best is to the known minimum (0).
    
    Parameters:
        mu: (2000,) array of GP means in objective units
        sigma: (2000,) array of GP std devs in objective units
        f_best: scalar, lowest observed value
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of finite scores (higher = better to evaluate)
    """
    start_time = time.time()
    
    # Clamp sigma to avoid division issues
    sigma = np.maximum(sigma, 1e-6)
    
    # Compute EI component
    z_ei = (f_best - mu) / sigma
    # Use numerically stable computation
    ei = (f_best - mu) * _safe_norm_cdf(z_ei) + sigma * norm.pdf(z_ei)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB component: mu - beta * sigma (lower is better, so we want low mu and high sigma)
    # For minimization, UCB score = mu - beta * sigma, but we want higher scores to be better
    # So we use: -mu + beta * sigma for the exploration part
    # Adaptive beta: larger when f_best is far from 0 (early exploration), smaller when close
    # f_best ranges from ~large (rastrigin can be ~20+) down to 0
    # Use a smooth schedule
    beta = 2.0 * np.exp(-f_best / 5.0) + 0.5
    
    # UCB score (higher is better for selection): -mu + beta * sigma
    ucb_score = -mu + beta * sigma
    
    # Adaptive weighting: blend EI and UCB
    # When f_best is large (far from optimum), weight more on UCB for exploration
    # When f_best is small (close to optimum), weight more on EI for exploitation
    # weight_ei goes from ~0.3 (early) to ~0.9 (late)
    weight_ei = 0.3 + 0.6 * (1.0 - np.exp(-f_best / 3.0))
    weight_ucb = 1.0 - weight_ei
    
    # Normalize EI and UCB to comparable scales
    # EI is in objective units, UCB score is also in objective units
    # But their magnitudes differ, so let's normalize by typical scale
    # Use a simple approach: scale EI by its max to [0,1] range, same for UCB
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    ucb_max = np.max(np.abs(ucb_score))
    if ucb_max > 1e-12:
        ucb_norm = ucb_score / ucb_max
    else:
        ucb_norm = ucb_score
    
    # Combined score before cost normalization
    combined = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Cost-aware normalization: divide by cost (higher cost = lower score)
    # cost = 1 + mean(x_scaled), so cost is in [1, 2]
    cost_safe = np.maximum(cost, 1e-6)
    score = combined / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard: if we've used too much time, return what we have
    elapsed = time.time() - start_time
    if elapsed > 280:
        return score
    
    return score
# EVOLVE-BLOCK-END
