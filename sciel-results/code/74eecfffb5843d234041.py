# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time for deadline guard
_START_TIME = time.time()
DEADLINE_MARGIN = 30.0  # Safety margin before 330s limit

def _elapsed():
    return time.time() - _START_TIME

def _expected_improvement(mu, sigma, f_best, xi=0.0):
    """Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Args:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value
        cost: (2000,) array of evaluation costs (1 + mean(x_scaled))
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're running low on time, return simple EI/cost quickly
    if _elapsed() > DEADLINE_MARGIN:
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / (cost + 1e-12)
        return np.where(np.isfinite(score), score, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Adaptive xi: small positive xi in late phase to focus exploitation
    f_scale = abs(f_best)
    if f_scale > 10.0:
        xi = 0.0
    elif f_scale > 1.0:
        xi = 0.01
    else:
        xi = 0.05
    
    # 1. Expected Improvement with adaptive xi, divided by log(cost) for smoother cost sensitivity
    ei = _expected_improvement(mu, sigma_safe, f_best, xi=xi)
    log_cost = np.log(cost + 1e-12)
    ei_cost = ei / (log_cost + 1e-12)
    
    # 2. Lower Confidence Bound for minimization: mu - beta*sigma
    #    Use adaptive beta: larger in early phase, smaller in late phase
    if f_scale > 10.0:
        beta = 2.5
    elif f_scale > 1.0:
        beta = 2.0
    else:
        beta = 1.5
    ucb = mu - beta * sigma_safe
    # Normalize: how much better than f_best could we potentially do
    ucb_norm = np.maximum(f_best - ucb, 0.0) / (abs(f_best) + 1.0)
    
    # 3. Cost-efficiency of exploration: high sigma / low cost
    cost_exp = sigma_safe / (cost + 1e-12)
    cost_exp_max = np.max(cost_exp) if len(cost_exp) > 0 else 1.0
    cost_exp_norm = cost_exp / (cost_exp_max + 1e-12)
    
    # Smooth adaptive weighting using sigmoid transitions
    # t1: transition from early to middle (f_scale around 10)
    # t2: transition from middle to late (f_scale around 1)
    t1 = 1.0 / (1.0 + np.exp(-(f_scale - 10.0) / 2.0))
    t2 = 1.0 / (1.0 + np.exp(-(f_scale - 1.0) / 0.5))
    
    # Early: w_ei=0.65, w_ucb=0.10, w_cost_exp=0.25
    # Middle: w_ei=0.50, w_ucb=0.20, w_cost_exp=0.30
    # Late: w_ei=0.40, w_ucb=0.25, w_cost_exp=0.35
    
    w_ei = 0.40 + 0.10 * t2 + 0.15 * t1 * (1.0 - t2)
    w_ucb = 0.25 - 0.05 * t2 - 0.10 * t1 * (1.0 - t2)
    w_cost_exp = 0.35 - 0.05 * t2 - 0.10 * t1 * (1.0 - t2)
    
    # Ensure weights sum to 1
    w_sum = w_ei + w_ucb + w_cost_exp
    w_ei /= w_sum
    w_ucb /= w_sum
    w_cost_exp /= w_sum
    
    score = w_ei * ei_cost + w_ucb * ucb_norm + w_cost_exp * cost_exp_norm
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
