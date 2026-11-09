"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: EI/cost with adaptive UCB exploration.
    
    mu, sigma, cost: shape (2000,)
    Returns: 2000 finite scores (higher = evaluate next)
    """
    t_start = time.time()
    DEADLINE = 320.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    n = mu.shape[0]
    
    # Handle non-finite inputs
    if not np.all(np.isfinite(mu)):
        mu = np.nan_to_num(mu, nan=0.0, posinf=1e10, neginf=-1e10)
    if not np.all(np.isfinite(sigma)):
        sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    if not np.all(np.isfinite(cost)):
        cost = np.nan_to_num(cost, nan=1.0, posinf=10.0, neginf=1.0)
    
    sigma = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute EI for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI
    ei_cost = ei / cost_safe
    
    # UCB exploration term for minimization
    # Adaptive beta: more exploration early, less as we converge
    # Use the spread of mu as a proxy for convergence
    mu_range = np.ptp(mu) if n > 1 else 1.0
    sigma_mean = np.mean(sigma)
    
    # Beta scales with uncertainty relative to the scale of the problem
    beta = 0.5 + 0.5 * np.clip(sigma_mean / (mu_range + 1e-9), 0, 1)
    
    # UCB score: lower mu is better, higher sigma encourages exploration
    ucb = -mu + beta * sigma
    
    # Normalize both to comparable scales
    ei_cost_mean = np.mean(ei_cost)
    ei_cost_std = np.std(ei_cost)
    if ei_cost_std > 1e-12:
        ei_cost_norm = (ei_cost - ei_cost_mean) / ei_cost_std
    else:
        ei_cost_norm = np.zeros_like(ei_cost)
    
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb)
    if ucb_std > 1e-12:
        ucb_norm = (ucb - ucb_mean) / ucb_std
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Blend EI/cost with UCB
    # Weight: more weight on EI/cost when it has clear signal, more on UCB when uncertain
    # The ratio of EI/cost std to UCB std tells us which signal is stronger
    signal_ratio = ei_cost_std / (ucb_std + 1e-12)
    # Clip weight to [0.3, 0.7] to always have some of both
    w_ei = np.clip(0.3 + 0.4 * signal_ratio / (1 + signal_ratio), 0.3, 0.7)
    
    scores = w_ei * ei_cost_norm + (1 - w_ei) * ucb_norm
    
    # Scale back
    scores = scores * (ei_cost_mean + 1e-9)
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline guard
    elapsed = time.time() - t_start
    if elapsed > DEADLINE - 5.0:
        pass
    
    return scores
