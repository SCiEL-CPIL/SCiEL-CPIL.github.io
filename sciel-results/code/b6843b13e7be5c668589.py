"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Blends Expected Improvement (EI) for exploitation with UCB for exploration,
    while accounting for evaluation cost. This helps explore uncertain regions
    early when f_best is high and EI alone gives zero scores to many candidates.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from the GP
    sigma : array-like, shape (2000,)
        Standard deviation predictions from the GP
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure no negative or zero costs
    cost = np.maximum(cost, 1e-9)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB: Upper Confidence Bound for exploration
    # beta controls exploration; scale it based on how much improvement is still possible
    mu_min = np.min(mu)
    improvement_remaining = max(f_best - mu_min, 1e-9)
    
    # Normalize EI and UCB components to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 1e-9 else 1.0
    ucb_raw = mu + 1.5 * sigma
    ucb_max = np.max(ucb_raw) if np.max(ucb_raw) > 1e-9 else 1.0
    
    # Cost-aware EI
    cost_aware_ei = ei / cost
    
    # Cost-aware UCB (lower is better, so we want high UCB = low mu + high sigma)
    # For minimization, UCB score = -(mu - beta*sigma) / cost = (beta*sigma - mu) / cost
    ucb_score = (1.5 * sigma - mu) / cost
    
    # Normalize both to [0, 1] range for blending
    ei_norm = cost_aware_ei / (ei_max / np.min(cost) + 1e-9) if ei_max > 1e-9 else 0.0
    ucb_norm = ucb_score / (np.max(ucb_score) + 1e-9) if np.max(ucb_score) > 1e-9 else 0.0
    
    # Clip to [0, 1]
    ei_norm = np.clip(ei_norm, 0.0, 1.0)
    ucb_norm = np.clip(ucb_norm, 0.0, 1.0)
    
    # Blend: weight EI more heavily when improvement is small (converging),
    # weight UCB more when improvement is large (exploring)
    # Use a smooth transition based on relative improvement
    rel_improvement = improvement_remaining / (f_best - mu_min + 1e-9)
    
    # Beta: higher when more improvement is possible
    beta_weight = 0.3 + 0.4 * min(rel_improvement, 1.0)
    ei_weight = 1.0 - beta_weight
    
    # Combined score
    base_score = ei_weight * cost_aware_ei + beta_weight * ucb_score
    
    # Check deadline
    if time.time() > deadline:
        return np.maximum(base_score, 0.0)
    
    # Ensure all values are finite
    base_score = np.where(np.isfinite(base_score), base_score, 0.0)
    
    # Ensure non-negative
    base_score = np.maximum(base_score, 0.0)
    
    return base_score
# EVOLVE-BLOCK-END
