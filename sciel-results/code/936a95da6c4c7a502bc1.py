import numpy as np
import time
from scipy.stats import norm

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _start_time, _call_count
    if _start_time is None:
        _start_time = time.time()
    
    _call_count += 1
    
    # Deadline guard
    if time.time() - _start_time > 325:
        # Fallback to EI if time is tight
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.where(np.isfinite(ei), ei, 0.0)
        return ei
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate EI
    ei = (f_best - mu) * norm.cdf((f_best - mu) / sigma_safe) + sigma_safe * norm.pdf((f_best - mu) / sigma_safe)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Calculate UCB with decaying beta
    # Early: high exploration, Late: high exploitation
    # 30 total evaluations, 5 initial, so 25 optimization steps
    # Beta decays from ~2.0 to ~0.5
    beta = 2.0 - 1.5 * min(_call_count / 25.0, 1.0)
    
    # UCB score (higher is better for maximization of score)
    # For minimization: ucb = mu - beta * sigma
    # But we want to return scores where higher means "evaluate next"
    # So we want to prefer low mu and high sigma
    # ucb_score = -mu + beta * sigma  (higher is better)
    ucb_score = -mu + beta * sigma_safe
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, -np.inf)
    
    # Normalize EI to be comparable to UCB scale
    # Use a simple normalization: divide by max EI if positive
    max_ei = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_normalized = ei / max_ei
    
    # Blend: early on rely more on UCB (exploration), later rely more on EI (exploitation)
    # Weight for EI increases with call count
    ei_weight = 0.3 + 0.7 * min(_call_count / 25.0, 1.0)
    ucb_weight = 1.0 - ei_weight
    
    # Combine
    # Scale UCB to be in a similar range as normalized EI
    # UCB values can be large, so let's normalize them too
    ucb_max = np.max(ucb_score) if np.max(ucb_score) > 0 else 1.0
    ucb_min = np.min(ucb_score)
    if ucb_max > ucb_min:
        ucb_normalized = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_normalized = np.zeros_like(ucb_score)
    
    combined = ei_weight * ei_normalized + ucb_weight * ucb_normalized
    
    # Ensure finite
    combined = np.where(np.isfinite(combined), combined, 0.0)
    
    return combined
