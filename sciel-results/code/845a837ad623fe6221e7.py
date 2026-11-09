"""
Cost-aware acquisition function for Bayesian optimization (rastrigin 2d).
Uses cost-aware UCB with adaptive exploration for robust multimodal optimization.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard constants
_START_TIME = time.time()
_DEADLINE_SECONDS = 285.0  # Safety margin before 300s limit

def _check_deadline():
    """Check if we are close to the time limit."""
    if time.time() - _START_TIME > _DEADLINE_SECONDS:
        return True
    return False

def _cost_aware_ucb(mu, sigma, f_best, cost, beta):
    """
    Cost-aware Upper Confidence Bound for minimization.
    UCB = -mu + beta * sigma
    Cost-aware: UCB / sqrt(cost)
    Using sqrt(cost) instead of cost for more balanced cost penalty.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # UCB for minimization: want low mu, high sigma
    ucb = -mu + beta * sigma_safe
    
    # Cost-aware normalization with sqrt for more balanced penalty
    scores = ucb / np.sqrt(cost_safe)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

def _expected_regret(mu, sigma, f_best):
    """
    Compute Expected Regret (ER) for minimization.
    ER = E[max(f_best - f(x), 0)]
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Standard normal CDF and PDF using scipy
    from scipy.stats import norm
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    er = diff * Phi + sigma_safe * phi
    er = np.maximum(er, 0.0)
    
    return er

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing rastrigin 2d.
    Uses adaptive cost-aware UCB with exploration-exploitation balance.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check deadline
    if _check_deadline():
        cost_safe = np.maximum(cost, 1e-9)
        return (-mu + 1.5 * sigma) / np.sqrt(cost_safe)
    
    # Adaptive beta for UCB:
    # Rastrigin is highly multimodal, so we need significant exploration
    # Start with high exploration, reduce as we find better solutions
    if f_best > 15.0:
        beta = 2.5
    elif f_best > 10.0:
        beta = 2.0
    elif f_best > 5.0:
        beta = 1.5
    elif f_best > 2.0:
        beta = 1.2
    elif f_best > 1.0:
        beta = 1.0
    else:
        beta = 0.8
    
    # Compute cost-aware UCB
    ucb_scores = _cost_aware_ucb(mu, sigma, f_best, cost, beta=beta)
    
    # Also compute cost-aware ER for blending
    er = _expected_regret(mu, sigma, f_best)
    cost_safe = np.maximum(cost, 1e-9)
    er_scores = er / np.sqrt(cost_safe)
    
    # Blend: more UCB for exploration (multimodal), more ER for exploitation
    if f_best > 10.0:
        blend = 0.3  # 30% ER, 70% UCB - heavy exploration
    elif f_best > 5.0:
        blend = 0.4
    elif f_best > 2.0:
        blend = 0.5
    elif f_best > 1.0:
        blend = 0.6
    else:
        blend = 0.7  # 70% ER, 30% UCB - more exploitation
    
    # Normalize both score sets
    ucb_max = np.max(ucb_scores)
    er_max = np.max(er_scores)
    
    if ucb_max > 1e-12:
        ucb_normalized = ucb_scores / ucb_max
    else:
        ucb_normalized = np.zeros_like(ucb_scores)
    
    if er_max > 1e-12:
        er_normalized = er_scores / er_max
    else:
        er_normalized = np.zeros_like(er_scores)
    
    # Blend
    scores = blend * er_normalized + (1.0 - blend) * ucb_normalized
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
