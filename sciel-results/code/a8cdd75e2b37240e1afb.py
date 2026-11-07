"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE_MARGIN = 15.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 3D.
    
    Uses an adaptive blend of Expected Improvement (EI) and UCB that shifts
    from exploration to exploitation as the median cost rises (proxy for
    number of evaluations), balancing the 30-cost budget across instances.
    
    Args:
        mu: Array of shape (N,) with GP mean predictions.
        sigma: Array of shape (N,) with GP standard deviations.
        f_best: The best observed function value so far.
        cost: Array of shape (N,) with evaluation costs for each candidate.
        
    Returns:
        Array of shape (N,) with acquisition scores (higher is better).
    """
    # Deadline guard
    if time.time() - _start_time > 300 - _DEADLINE_MARGIN:
        # Return a simple fallback: negative mu normalized by cost
        mu = np.asarray(mu)
        cost = np.asarray(cost)
        cost_safe = np.maximum(cost, 1e-9)
        return np.nan_to_num(-mu / cost_safe, nan=0.0, posinf=0.0, neginf=0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure finite
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute EI for minimization
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB for minimization: -mu + kappa * sigma
    # Adaptive kappa: start high (exploration), decrease as we gather info
    # Use median cost as proxy for progress (higher median cost means more evals done)
    median_cost = np.median(cost)
    # median_cost ranges from ~1.5 (early) to ~2.0 (late)
    # Map to kappa from 4.0 down to 1.5
    kappa = 4.0 - 2.5 * np.clip((median_cost - 1.5) / 0.5, 0, 1)
    ucb = -mu + kappa * sigma_safe
    
    # Adaptive weighting: shift from UCB to EI as we progress
    # Early: more UCB (exploration), Late: more EI (exploitation)
    progress = np.clip((median_cost - 1.5) / 0.5, 0, 1)
    w_ei = 0.3 + 0.5 * progress  # 0.3 to 0.8
    w_ucb = 1.0 - w_ei           # 0.7 to 0.2
    
    # Normalize EI and UCB to comparable scales
    # EI is in objective units, UCB is in objective units too
    # But their magnitudes differ. Let's scale EI by the range of mu
    mu_range = np.ptp(mu) if np.ptp(mu) > 1e-9 else 1.0
    ei_scaled = ei / mu_range
    ucb_scaled = ucb / mu_range
    
    score_numerator = w_ei * ei_scaled + w_ucb * ucb_scaled
    
    # Normalize by cost
    scores = score_numerator / cost_safe
    
    # Handle any NaNs or Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
