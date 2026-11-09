import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 10.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid EI + UCB approach with adaptive blending.
    
    The blend weight is determined by the exploration-exploitation tradeoff:
    - Early on (high uncertainty relative to best), UCB dominates for exploration.
    - Later (low uncertainty relative to best), EI dominates for exploitation.
    
    The UCB component uses a time-varying beta that decreases as we get closer
    to the deadline, encouraging exploitation in later steps.
    
    Args:
        mu: Mean of the GP posterior at candidates (shape 2000,).
        sigma: Std dev of the GP posterior at candidates (shape 2000,).
        f_best: Current best function value (minimum).
        
    Returns:
        scores: Array of shape 2000, higher is better.
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Fallback to pure EI if too close to deadline
        z = (f_best - mu) / (sigma + 1e-9)
        scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    n = len(mu)
    
    # Ensure sigma is positive to avoid errors
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute EI (Expected Improvement)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute UCB (Upper Confidence Bound)
    # UCB = mu + beta * sigma, where beta controls exploration
    # Use a time-varying beta: start high for exploration, decrease for exploitation
    # The fraction of time remaining can guide beta
    time_remaining_fraction = max(0.0, (_DEADLINE - elapsed) / _DEADLINE)
    
    # Beta decreases from ~3.0 early to ~0.5 late
    # This encourages more exploration early and more exploitation later
    beta = 0.5 + 2.5 * time_remaining_fraction
    
    ucb = mu + beta * sigma_safe
    
    # Normalize EI and UCB to comparable scales for blending
    # EI is already in objective units (improvement over f_best)
    # UCB is in objective units (predicted value + exploration bonus)
    # We want higher scores to mean "evaluate next"
    # For minimization: lower predicted value is better
    # EI: higher = more expected improvement = better
    # UCB: lower mu + higher sigma = more exploration potential
    # But UCB as defined above gives higher values for more uncertain/lower mean
    # We need to invert UCB for minimization: score_ucb = -ucb
    # Actually, let's think about this more carefully:
    # - EI directly measures expected improvement (higher is better)
    # - For UCB in minimization, we want to explore where mu is low and sigma is high
    #   So score_ucb = -mu + beta * sigma (higher when mu is low and sigma is high)
    
    score_ucb = -mu + beta * sigma_safe
    
    # Normalize both to zero mean, unit variance for blending
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-9
    ei_norm = (ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(score_ucb)
    ucb_std = np.std(score_ucb) + 1e-9
    ucb_norm = (score_ucb - ucb_mean) / ucb_std
    
    # Adaptive blend weight:
    # When uncertainty is high relative to the range of f_best, use more UCB
    # When uncertainty is low, use more EI
    # Measure the relative uncertainty: mean(sigma) / (|f_best| + 1)
    rel_uncertainty = np.mean(sigma_safe) / (abs(f_best) + 1.0)
    
    # Blend weight for UCB: higher when relative uncertainty is high
    # Use a smooth sigmoid-like transition
    # w_ucb = 0.5 + 0.5 * tanh(5 * (rel_uncertainty - 0.5))
    # This gives ~0 when rel_uncertainty is low, ~1 when high
    w_ucb = 0.5 + 0.5 * np.tanh(5.0 * (rel_uncertainty - 0.3))
    w_ei = 1.0 - w_ucb
    
    # Combine
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
