import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_TIME_LIMIT = 295.0  # Leave 5s margin before 300s kill
_call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid of Thompson Sampling and Expected Improvement.
    
    For minimization:
    - Thompson Sampling (TS) provides stochastic exploration by sampling from the
      posterior predictive distribution, helping escape local minima in the
      multimodal Weierstrass landscape.
    - Expected Improvement (EI) provides a stable exploitation baseline.
    - The weights adapt over time: high exploration early, shifting to exploitation
      later.
    
    Args:
        mu: Array of mean predictions (2000,).
        sigma: Array of standard deviation predictions (2000,).
        f_best: The best (lowest) function value observed so far.
        
    Returns:
        Array of scores (2000,). Higher scores indicate better candidates.
    """
    global _call_count
    
    # Deadline guard: if time is running out, fall back to a fast deterministic method
    if time.time() - _start_time > _TIME_LIMIT:
        # Fast fallback: use EI only (deterministic, fast)
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        z_clipped = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    _call_count += 1
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # 1. Expected Improvement (EI) component
    # Standard EI for minimization
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # 2. Thompson Sampling (TS) component
    # Sample from the posterior predictive distribution
    sample = np.random.normal(loc=mu, scale=sigma_safe)
    
    # The TS term: we want to maximize the probability of finding a better point.
    # A common approach is to use the negative of the sample (since we minimize).
    # But to combine with EI (which is non-negative), we can use a different formulation.
    # Let's use: score = w_ei * EI + w_ts * (f_best - sample)
    # If sample < f_best, (f_best - sample) > 0, boosting exploration.
    # If sample > f_best, (f_best - sample) < 0, penalizing it.
    
    ts_term = f_best - sample
    
    # Adaptive weights: Start with high exploration, decay to low exploration
    # Total evaluations ~ 30. We are at call_count (1-based for acquisition calls).
    # There are 25 acquisition calls total (30 total - 5 initial).
    total_calls = 25
    progress = min(_call_count / total_calls, 1.0)
    
    # Exploration weight: starts at 1.0, ends at 0.1
    # Using a smooth decay
    w_ts = 0.1 + 0.9 * (1.0 - progress)
    
    # Exploitation weight: constant 1.0 for EI
    w_ei = 1.0
    
    # Combine
    scores = w_ei * ei + w_ts * ts_term
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
