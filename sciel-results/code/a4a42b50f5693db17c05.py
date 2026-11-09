import numpy as np
from scipy.stats import norm
import time

# Global state for adaptive parameter tuning
_state = {
    'best_f': np.inf,
    'f_history': [],
    'n_calls': 0,
    'start_time': None,
    'sigma_history': []
}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement with adaptive
    exploration-exploitation balance and cost normalization.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - best observed function value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = prefer)
    """
    global _state
    
    # Initialize state
    if _state['start_time'] is None:
        _state['start_time'] = time.time()
    
    _state['n_calls'] += 1
    _state['f_history'].append(f_best)
    _state['sigma_history'].append(np.mean(sigma))
    
    elapsed = time.time() - _state['start_time']
    
    # Safety: if we're close to timeout, use a fast fallback
    if elapsed > 280:
        sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
        cost_safe = np.where(cost > 1e-9, cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Handle edge cases
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    cost_safe = np.where(cost > 1e-9, cost, 1e-9)
    
    # Track progress
    if f_best < _state['best_f']:
        _state['best_f'] = f_best
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute LCB (Lower Confidence Bound) for exploration
    # Adaptive beta based on uncertainty relative to f_best scale
    sigma_95 = np.percentile(sigma_safe, 95)
    sigma_mean = np.mean(sigma_safe)
    
    # Beta adapts: larger when uncertainty is high relative to best value
    # Use a smooth transition
    f_scale = max(abs(f_best), 1e-3)
    beta = 1.0 + 1.5 * (sigma_95 / (f_scale + sigma_95 + 1e-9))
    
    lcb = mu - beta * sigma_safe  # For minimization, lower is better
    
    # Blend EI and LCB
    # Early on: more exploration (LCB), later: more exploitation (EI)
    n_calls = _state['n_calls']
    
    # Progress-based weight: start with more exploration, shift to exploitation
    # Use a smooth transition over ~20 calls
    progress = min(n_calls / 20.0, 1.0)
    w_ei = 0.3 + 0.5 * progress  # EI weight: 0.3 -> 0.8
    w_lcb = 1.0 - w_ei           # LCB weight: 0.7 -> 0.2
    
    # Normalize both to comparable scales before blending
    # EI is always >= 0, LCB can be negative
    ei_norm = ei
    # Shift LCB so it's centered around 0 with similar scale to EI
    lcb_shifted = lcb - np.median(lcb)
    lcb_norm = lcb_shifted * (np.std(ei_norm) / (np.std(lcb_shifted) + 1e-9))
    
    blended = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Cost-aware scoring: divide by cost, but use a softer penalty
    # Cost is in [1, 2] range (1 + mean of 4 coords in [0,1])
    # Use sqrt(cost) as penalty to be less aggressive than linear
    cost_penalty = np.sqrt(cost_safe)
    
    scores = blended / cost_penalty
    
    # Add a small exploration bonus for high-sigma points to encourage diversity
    # This helps escape local optima
    sigma_rank = np.argsort(sigma_safe)  # Low to high sigma
    exploration_bonus = 0.02 * (np.argsort(sigma_rank) / len(sigma_rank)) * sigma_95
    scores += exploration_bonus / cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
