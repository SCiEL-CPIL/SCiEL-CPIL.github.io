import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    if time.time() - _start_time > _DEADLINE:
        # Fallback: Pure EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # UCB (Upper Confidence Bound) for minimization:
    # We want to minimize, so we look for low mu and high sigma (uncertainty).
    # Standard UCB for minimization: mu - beta * sigma? 
    # No, for minimization, we want to find points where the lower bound is low.
    # The lower bound is mu - beta * sigma.
    # So we maximize (mu - beta * sigma) ? No, we want to minimize the function.
    # We want to select x that has a high probability of being good (low value).
    # UCB for minimization is often defined as: mu(x) - beta * sigma(x).
    # We want to maximize this score? 
    # If mu is low and sigma is high, mu - beta*sigma is very low.
    # So we want to MINIMIZE (mu - beta * sigma)?
    # Or MAXIMIZE (beta * sigma - mu)?
    # Let's check standard definitions.
    # For maximization: UCB = mu + beta * sigma. Maximize UCB.
    # For minimization: LCB = mu - beta * sigma. Minimize LCB.
    # Or equivalently, Maximize (beta * sigma - mu).
    
    beta = 2.5
    # Score = beta * sigma - mu
    # This score is high when mu is low (good) and sigma is high (exploration).
    scores = beta * sigma_safe - mu
    
    # Cost-aware: divide by cost to prefer cheaper points
    scores = scores / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
