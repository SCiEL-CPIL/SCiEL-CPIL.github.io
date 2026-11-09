import numpy as np
from scipy.stats import norm
import time

_DEADLINE_START = time.time()
_TIME_LIMIT = 310.0  # Safety margin before 330s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _DEADLINE_START)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (hartmann 3d).
    Blends Expected Improvement (EI), Probability of Improvement (PI),
    and a normalized uncertainty bonus with cost-awareness.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Deadline guard
    if _time_left() < 1.0:
        sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-10)
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-10)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure inputs are properly shaped and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (stable computation)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Normalize EI and PI to [0, 1] range for blending
    ei_max = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    pi_max = np.max(pi) if np.max(pi) > 1e-12 else 1.0
    
    ei_norm = ei / ei_max
    pi_norm = pi / pi_max
    
    # Normalized uncertainty bonus: encourages exploration in high-sigma regions
    # This provides a smooth exploration signal less sensitive to extreme values than UCB
    max_sigma = np.max(sigma_safe)
    if max_sigma > 1e-12:
        uncertainty_norm = sigma_safe / max_sigma
    else:
        uncertainty_norm = np.zeros_like(sigma_safe)
    
    # Blend: 0.5 EI (exploitation), 0.4 PI (balanced), 0.1 uncertainty (exploration)
    blended = 0.5 * ei_norm + 0.4 * pi_norm + 0.1 * uncertainty_norm
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    scores = blended / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
