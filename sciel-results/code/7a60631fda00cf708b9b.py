"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global timer for deadline guard
_start_time = time.time()
TIME_LIMIT = 290  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best):
    """
    Hybrid Thompson Sampling + LCB acquisition function for minimization.
    
    Combines the stochastic exploration of Thompson Sampling (which showed
    strong performance in small-budget multimodal optimization) with a
    deterministic Lower Confidence Bound (LCB) component to ensure
    structured exploration even when the random sample is unlucky.
    
    Args:
        mu: Array of posterior means (2000,)
        sigma: Array of posterior standard deviations (2000,)
        f_best: Minimum observed value
        
    Returns:
        Array of scores (2000,), higher is better.
    """
    # Deadline guard
    if time.time() - _start_time > TIME_LIMIT:
        # If we are running out of time, fall back to a simple deterministic score
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 2.0 * sigma_safe

    # Ensure numerical stability
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling component:
    # Draw a single random sample from the posterior
    eps = np.random.normal(0, 1, size=mu.shape)
    ts_sample = mu + sigma_safe * eps
    # We want to minimize ts_sample, so score is -ts_sample
    ts_score = -ts_sample
    
    # LCB component:
    # Lower Confidence Bound for minimization: mu - beta * sigma
    # We want to minimize this, so score is -(mu - beta * sigma) = beta * sigma - mu
    beta = 2.0
    lcb_score = beta * sigma_safe - mu
    
    # Hybrid: combine Thompson Sampling with LCB
    # Weight Thompson Sampling heavily (it's the stronger performer) but
    # add LCB as a structured safety net to prevent bad random luck
    alpha = 0.7  # Weight for Thompson Sampling
    scores = alpha * ts_score + (1 - alpha) * lcb_score
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    return scores
# EVOLVE-BLOCK-END
