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
    
    # 1. Expected Improvement per unit cost
    ei = _expected_improvement(mu, sigma_safe, f_best, xi=0.0)
    ei_cost = ei / (cost + 1e-12)
    
    # 2. Lower Confidence Bound for minimization: mu - beta*sigma
    #    Higher (less negative) is better for minimization
    beta = 2.0
    ucb = mu - beta * sigma_safe
    # Normalize by scale of f_best to make comparable
    ucb_norm = np.maximum(f_best - ucb, 0.0) / (abs(f_best) + 1.0)
    
    # 3. Probability of Improvement
    z_pi = (f_best - mu) / sigma_safe
    pi = norm.cdf(z_pi)
    
    # 4. Cost-efficiency of exploration: high sigma / low cost
    cost_exp = sigma_safe / (cost + 1e-12)
    cost_exp_max = np.max(cost_exp) if len(cost_exp) > 0 else 1.0
    cost_exp_norm = cost_exp / (cost_exp_max + 1e-12)
    
    # 5. GP uncertainty normalized
    sigma_max = np.max(sigma_safe) if len(sigma_safe) > 0 else 1.0
    sigma_norm = sigma_safe / (sigma_max + 1e-12)
    
    # Thompson sampling: add stochastic perturbation to break deterministic exploitation
    # Use a seeded random number generator based on f_best and cost to ensure reproducibility
    # within a single call but vary across calls
    seed_val = int(abs(hash((f_best, float(np.mean(cost))))) % 2**31)
    rng = np.random.default_rng(seed_val)
    
    # Scale of perturbation: larger when sigma is high, smaller when f_best is close to optimum
    f_scale = abs(f_best)
    if f_scale > 10.0:
        pert_scale = 0.1
    elif f_scale > 1.0:
        pert_scale = 0.05
    else:
        pert_scale = 0.02
    
    # Add Gaussian noise scaled by local sigma to encourage diverse exploration
    perturbation = rng.normal(0, 1, size=len(mu)) * pert_scale * sigma_safe
    perturbation = perturbation / (cost + 1e-12)  # Cost-aware perturbation
    
    # Adaptive weighting: as f_best approaches 0, rely more on exploration
    if f_scale > 10.0:
        # Early phase: strong exploitation
        w_ei = 0.55
        w_ucb = 0.15
        w_pi = 0.05
        w_cost_exp = 0.25
    elif f_scale > 1.0:
        # Middle phase: balanced
        w_ei = 0.40
        w_ucb = 0.20
        w_pi = 0.05
        w_cost_exp = 0.35
    else:
        # Late phase: exploration focused
        w_ei = 0.25
        w_ucb = 0.25
        w_pi = 0.05
        w_cost_exp = 0.45
    
    score = w_ei * ei_cost + w_ucb * ucb_norm + w_pi * pi + w_cost_exp * cost_exp_norm + perturbation
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
