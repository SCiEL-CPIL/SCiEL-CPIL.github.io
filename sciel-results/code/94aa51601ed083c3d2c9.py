"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track elapsed time to ensure we finish well within the 300s limit
_START_TIME = time.time()
DEADLINE_LIMIT = 300.0
SAFETY_MARGIN = 10.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining_budget():
    return DEADLINE_LIMIT - _elapsed() - SAFETY_MARGIN

def _expected_improvement(mu, sigma, f_best, xi=0.0):
    """
    Standard Expected Improvement for minimization.
    EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu - xi) / sigma
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _upper_confidence_bound(mu, sigma, beta):
    """
    Upper Confidence Bound for minimization: mu - beta * sigma
    """
    return mu - beta * sigma

def _probability_of_improvement(mu, sigma, f_best):
    """
    Probability of Improvement for minimization: P(f(x) < f_best).
    PI = Phi((f_best - mu) / sigma)
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    return norm.cdf(z)

def _cost_adjusted_ei(mu, sigma, f_best, cost, xi=0.01):
    """
    Cost-adjusted Expected Improvement: EI / cost
    """
    ei = _expected_improvement(mu, sigma, f_best, xi=xi)
    cost_safe = np.maximum(cost, 1e-12)
    return ei / cost_safe

def _ucb_cost_aware(mu, sigma, f_best, cost, beta):
    """
    Cost-aware UCB: (f_best - UCB) / cost
    """
    ucb = _upper_confidence_bound(mu, sigma, beta)
    improvement = np.maximum(f_best - ucb, 0.0)
    cost_safe = np.maximum(cost, 1e-12)
    return improvement / cost_safe

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a mixture of cost-adjusted EI, cost-aware UCB, and cost-aware PI
    with adaptive weights based on the current best value.
    
    Args:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Scale factor for normalization
    f_best_scale = abs(f_best) + 1.0
    
    # Progress indicator: early (f_best large) vs late (f_best small)
    # Use a smooth transition
    progress = 1.0 / (1.0 + 0.5 * f_best_scale)
    explore_weight = 1.0 - progress
    exploit_weight = progress
    
    # 1. Cost-adjusted EI with adaptive xi
    xi = 0.01 * f_best_scale
    cai = _cost_adjusted_ei(mu, sigma, f_best, cost, xi=xi)
    
    # 2. Cost-aware UCB with adaptive beta
    beta = 1.0 + 1.0 * explore_weight
    cu = _ucb_cost_aware(mu, sigma, f_best, cost, beta=beta)
    
    # 3. Cost-aware PI
    pi = _probability_of_improvement(mu, sigma, f_best)
    cp = pi / cost
    
    # Normalize each component to [0, 1] range
    def _normalize(arr):
        arr_max = np.max(arr)
        if arr_max > 1e-12:
            return arr / arr_max
        return np.zeros_like(arr)
    
    cai_norm = _normalize(cai)
    cu_norm = _normalize(cu)
    cp_norm = _normalize(cp)
    
    # Adaptive weights: blend between exploration and exploitation
    w_cai = 0.4 * exploit_weight + 0.3 * explore_weight
    w_cu = 0.3 * explore_weight + 0.2 * exploit_weight
    w_cp = 0.15 * exploit_weight + 0.1 * explore_weight
    
    # Final score
    score = w_cai * cai_norm + w_cu * cu_norm + w_cp * cp_norm
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
