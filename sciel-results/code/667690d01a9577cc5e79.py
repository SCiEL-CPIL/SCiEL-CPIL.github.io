# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 320s as safety margin
_DEADLINE = 320.0
_START_TIME = time.time()


def _check_deadline():
    """Return True if we should abort and return a safe fallback."""
    return (time.time() - _START_TIME) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Styblinski-Tang 2D.
    
    Parameters:
    -----------
    mu : array, shape (2000,)
        GP mean predictions in objective units.
    sigma : array, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed function value so far.
    cost : array, shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x_scaled_to_01).
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = more promising to evaluate).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Deadline guard: return a safe finite fallback
    if _check_deadline():
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        # Simple exploitation: prefer low mu, penalize high cost
        return -mu / cost_safe
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB for minimization: we want to maximize -mu + kappa * sigma
    # Use adaptive kappa based on progress through budget
    # Progress: use the mean cost as a proxy for how far we are in the budget
    mean_cost = float(np.mean(cost_safe))
    # Budget is ~30 total cost; progress in [0, 1]
    progress = np.clip(mean_cost / 30.0, 0.0, 1.0)
    
    # Adaptive kappa: start with more exploration, shift to exploitation
    # Early: kappa ~ 2.0 (explore), Late: kappa ~ 0.5 (exploit)
    kappa = 2.0 - 1.5 * progress
    kappa = np.clip(kappa, 0.3, 2.5)
    
    ucb = -mu + kappa * sigma_safe
    
    # Normalize EI and UCB to comparable scales
    # Use the spread of (f_best - mu) as a reference for EI scale
    diff = np.maximum(f_best - mu, 0.0)
    ref_ei_scale = max(float(np.mean(diff)), 1e-6)
    
    # For UCB, use the std of mu as a reference scale
    ref_ucb_scale = max(float(np.std(mu)), 1e-6)
    
    # Normalize both to have similar magnitude
    ei_norm = ei / (ref_ei_scale + 1e-9)
    ucb_norm = ucb / (ref_ucb_scale + 1e-9)
    
    # Adaptive blend: weight based on progress
    # Early: more UCB (exploration), Late: more EI (exploitation)
    # Use a smooth transition
    w_ei = 0.3 + 0.7 * (progress ** 1.5)
    w_ucb = 1.0 - w_ei
    
    # Combine
    blended = w_ei * ei_norm + w_ucb * ucb_norm
    blended = np.maximum(blended, 0.0)
    
    # Cost-aware: divide by cost to favor cheap evaluations
    # But don't over-penalize; use a softer cost scaling
    # cost ranges from ~1 to ~2, so cost_safe is in [1, 2]
    # Use cost^0.5 to soften the penalty
    cost_factor = np.sqrt(cost_safe)
    score = blended / cost_factor
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
