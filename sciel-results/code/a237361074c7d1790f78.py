import numpy as np
from scipy.stats import norm
import time

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Shekel 4D.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed function value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    global _start_time, _call_count
    
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Safety fallback: simple cost-aware exploitation
        return -mu / (np.maximum(cost, 1e-9))
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-9, 1e6)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Thompson Sampling Heuristic
    # Sample a single function value from the GP posterior for each candidate
    # y ~ N(mu, sigma^2)
    # We use standard normal samples and transform them
    z_sample = np.random.standard_normal(len(mu))
    y_sample = mu + sigma * z_sample
    
    # Calculate the improvement of the sampled value over the current best
    # We want to maximize the improvement per unit cost
    improvement_sample = np.maximum(0.0, f_best - y_sample)
    
    # Cost-aware score
    scores = improvement_sample / cost_safe
    
    # To ensure stability and some exploitation even if samples are poor,
    # we can add a small term based on the mean prediction if it is promising.
    # However, the directive suggests replacing the score. 
    # A pure TS score might be 0 everywhere if f_best is very good and GP is accurate.
    # In that case, all scores are 0, leading to arbitrary selection.
    # To handle this, we can add a tiny epsilon or rely on the fact that 
    # if GP is accurate, mu should be close to f_best, and sigma small.
    # If mu < f_best, improvement is likely positive.
    
    # If all scores are 0, we might want to fall back to exploitation (lowest mu)
    # or exploration (highest sigma).
    # Let's check if any score is > 0.
    if np.max(scores) == 0.0:
        # Fallback: Exploit the best predicted mean, adjusted for cost
        # This happens if the GP predicts no point can beat f_best with the sampled noise
        # or if f_best is extremely low.
        scores = -mu / cost_safe
        
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
