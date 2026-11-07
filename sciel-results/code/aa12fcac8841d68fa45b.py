"""Cost-aware acquisition using PI + UCB with adaptive cost penalty."""
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0  # Safety margin before 300s kill

# Track iteration count to adapt exploration over time
_iter_count = 0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Probability of Improvement + UCB exploration,
    weighted by an adaptive cost penalty that encourages cheap evaluations.
    
    Args:
        mu: array shape (2000,) - GP mean predictions in objective units
        sigma: array shape (2000,) - GP std dev predictions in objective units
        f_best: float - lowest observed value
        cost: array shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array shape (2000,) - acquisition scores (higher = evaluate next)
    """
    global _iter_count
    _iter_count += 1
    
    # Time guard: if we're running out of time, return a simple fallback
    if time.time() > _DEADLINE:
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-8)
        mu_arr = np.asarray(mu, dtype=np.float64)
        return (-mu_arr) / cost_safe
    
    # Ensure all inputs are finite and properly shaped
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive and finite
    cost_safe = np.maximum(cost, 1e-8)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)
    
    # Ensure mu and sigma are finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    
    # Adaptive exploration: start with more exploration, decay over iterations
    # Beta controls UCB weight; higher early on for exploration
    beta = 1.5 * np.exp(-0.05 * _iter_count) + 0.3
    
    # Probability of Improvement for minimization:
    # PI(x) = Phi((f_best - mu) / sigma)
    diff = f_best - mu
    z = diff / sigma_safe
    z = np.clip(z, -10, 10)
    
    pi = norm.cdf(z)
    
    # UCB component: mu - beta * sigma (lower is better, so we negate for score)
    ucb = -(mu + beta * sigma_safe)
    
    # Combine: weight PI heavily when close to best, UCB for exploration
    # Normalize PI to [0, 1] range and blend
    # Score = alpha * PI + (1 - alpha) * UCB_normalized
    # UCB can be negative; normalize by scaling
    ucb_max = np.max(ucb) if len(ucb) > 0 else 0.0
    ucb_min = np.min(ucb) if len(ucb) > 0 else 0.0
    ucb_range = ucb_max - ucb_min
    if ucb_range < 1e-10:
        ucb_norm = np.zeros_like(ucb)
    else:
        ucb_norm = (ucb - ucb_min) / ucb_range  # [0, 1], higher = better (more negative mu)
    
    # Blend: PI gives probability of improvement, UCB gives exploration
    alpha = 0.6  # weight on PI
    combined = alpha * pi + (1.0 - alpha) * ucb_norm
    
    # Cost-aware: penalize expensive points more heavily
    # Use a power penalty: cost^gamma, with gamma > 1 to strongly prefer cheap points
    # Cost ranges from 1 (at x=0) to 3 (at corners)
    gamma = 2.0  # stronger cost penalty
    cost_penalty = np.power(cost_safe, gamma)
    
    score = combined / cost_penalty
    
    # Ensure output is finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
