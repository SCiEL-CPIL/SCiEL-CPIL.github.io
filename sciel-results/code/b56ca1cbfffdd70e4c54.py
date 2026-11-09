"""Cost-aware acquisition: EI/UCB blend with power-based cost penalty and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0  # 5s safety margin before 300s hard limit

def _elapsed_ok():
    """Return True if we have at least ~3s margin before the hard limit."""
    return (time.time() < _DEADLINE - 3.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization.
    
    mu, sigma, f_best, cost: each shape (2000,)
    f_best: scalar (lowest observed value)
    cost: per-candidate cost = 1 + mean(x_scaled)
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Deadline guard: if we're running out of time, return a simple fallback
    if not _elapsed_ok():
        # Simple fallback: just use -mu / cost to finish quickly
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)
        mu_arr = np.asarray(mu, dtype=np.float64)
        fallback = -mu_arr / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Robust sigma: ensure it's positive and not too small
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    # Handle numerical stability: clip z to avoid overflow in cdf/pdf
    z_clipped = np.clip(z, -50, 50)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # UCB for minimization: -mu + kappa * sigma
    kappa = 2.0
    ucb = -mu + kappa * sigma_safe
    
    # Blend EI and UCB with fixed weights
    w_ei = 0.6
    w_ucb = 0.4
    blended = w_ei * ei + w_ucb * ucb
    
    # Power-based cost penalty: cost in [1,2], use exponent 0.5 for mild penalty
    # This preserves relative ranking better than linear division
    cost_penalty = np.power(cost_safe, 0.5)
    score = blended / cost_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
