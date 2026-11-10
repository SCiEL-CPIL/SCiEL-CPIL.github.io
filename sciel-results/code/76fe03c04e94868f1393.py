"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s hard kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware hybrid acquisition: combines EI/cost (exploitation) with
    UCB/cost (exploration) and an uncertainty-weighted bonus.
    
    This is a multi-criterion acquisition that balances:
    1. Expected Improvement per cost (exploit known good regions)
    2. Upper Confidence Bound per cost (explore uncertain regions)
    3. A small entropy bonus for high-uncertainty areas
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far (lower is better).
    cost : np.ndarray, shape (2000,)
        Evaluation cost for each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = better to evaluate next).
    """
    # Time guard: if we're running low on time, fall back to fast EI/cost
    if time.time() - _start_time > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-12)
        cost_safe = np.maximum(cost, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Numerical stability
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # --- Component 1: Expected Improvement per cost ---
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50, 50)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    ei_score = ei / cost_safe
    
    # --- Component 2: UCB per cost ---
    # UCB = mu + beta * sigma (for minimization, we want low mu + high sigma)
    # Scale beta based on the scale of the problem
    # Use a moderate exploration parameter
    beta = 1.0
    ucb = mu + beta * sigma_safe
    # For minimization, lower UCB is better, so negate
    ucb_score = -ucb / cost_safe
    
    # --- Component 3: Uncertainty bonus ---
    # Reward high sigma regions with a small bonus to encourage exploration
    # Normalize sigma to [0, 1] range for this batch
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min
    if sigma_range > 1e-12:
        sigma_norm = (sigma_safe - sigma_min) / sigma_range
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    # Small bonus for high uncertainty
    unc_bonus = 0.1 * sigma_norm / cost_safe
    
    # --- Combine components ---
    # Normalize each component to have comparable scales
    ei_mean = np.mean(ei_score)
    ucb_mean = np.mean(np.abs(ucb_score))
    unc_mean = np.mean(unc_bonus)
    
    # Adaptive weights: shift from exploitation to exploration as f_best approaches minimum
    # Use a heuristic: if EI is small relative to sigma, favor exploration more
    ei_scale = np.mean(sigma_safe)
    if ei_scale > 1e-12:
        ei_ratio = np.mean(ei) / ei_scale
    else:
        ei_ratio = 0.0
    
    # Weights: start with more exploitation, shift to exploration when EI is low
    w_ei = 0.6 + 0.2 * np.clip(ei_ratio, 0, 1)
    w_ucb = 0.3 + 0.1 * (1 - np.clip(ei_ratio, 0, 1))
    w_unc = 0.1
    
    score = w_ei * ei_score + w_ucb * ucb_score + w_unc * unc_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
