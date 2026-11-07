"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline tracking
_acquisition_start_time = None
_DEADLINE_SECONDS = 320.0  # Safety margin before 330s kill

def _get_elapsed_time():
    global _acquisition_start_time
    if _acquisition_start_time is None:
        _acquisition_start_time = time.time()
    return time.time() - _acquisition_start_time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of GP mean predictions (shape 2000,)
    - sigma: array of GP standard deviations (shape 2000,)
    - f_best: lowest observed function value (scalar)
    - cost: array of evaluation costs (shape 2000,)
    
    Returns:
    - scores: array of acquisition values (shape 2000,), higher = evaluate next
    """
    global _acquisition_start_time
    
    # Deadline guard
    if _get_elapsed_time() > _DEADLINE_SECONDS:
        return np.zeros(2000, dtype=np.float64)
    
    # Ensure proper types and shapes
    mu = np.asarray(mu, dtype=np.float64).flatten()
    sigma = np.asarray(sigma, dtype=np.float64).flatten()
    cost = np.asarray(cost, dtype=np.float64).flatten()
    
    # Safety: ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring using a combination of:
    # 1. EI per unit cost (primary)
    # 2. Uncertainty-weighted exploration bonus
    # 3. Cost efficiency bonus
    
    # Primary: EI normalized by cost
    ei_per_cost = ei / cost
    
    # Secondary: exploration bonus for high uncertainty, penalized by cost
    # Use log-scale to avoid dominance by extreme uncertainty
    uncertainty_bonus = np.log1p(sigma) / cost
    
    # Tertiary: reward low-cost points when EI is small
    # This helps in early stages or when improvement is uncertain
    cost_mean = np.mean(cost)
    cost_std = np.std(cost) + 1e-9
    cost_z = (cost - cost_mean) / cost_std
    low_cost_bonus = 0.1 * np.maximum(0, -cost_z) / cost
    
    # Combine with adaptive weights based on the scale of EI
    ei_scale = np.mean(ei) + 1e-9
    ei_weight = min(1.0, ei_scale / 0.1)  # Scale EI weight based on typical improvement
    exp_weight = 0.3 * (1.0 - ei_weight) + 0.1  # Reduce exploration weight as EI grows
    
    scores = ei_weight * ei_per_cost + exp_weight * uncertainty_bonus + low_cost_bonus
    
    # Normalize scores to [0, 1] range for stability
    scores_min = np.min(scores)
    scores_max = np.max(scores)
    if scores_max - scores_min > 1e-12:
        scores = (scores - scores_min) / (scores_max - scores_min)
    else:
        scores = np.ones_like(scores) * 0.5
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
