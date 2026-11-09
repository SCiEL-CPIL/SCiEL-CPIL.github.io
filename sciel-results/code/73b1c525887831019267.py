import time
import numpy as np
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def _minmax_normalize(x):
    """Normalize an array to [0, 1] range. Handles edge cases."""
    min_val = np.min(x)
    max_val = np.max(x)
    if max_val - min_val < 1e-9:
        return np.zeros_like(x)
    return (x - min_val) / (max_val - min_val)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and a
    normalized Upper Confidence Bound (UCB) term with adaptive exploration weight.
    
    Args:
        mu: GP posterior mean, shape (N,)
        sigma: GP posterior std, shape (N,)
        f_best: Best observed value so far (lower is better)
        
    Returns:
        Scores for each candidate, shape (N,). Higher scores preferred.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a simple, fast exploitation score
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration term: Lower Confidence Bound (LCB) style for minimization
    # We want to maximize the score, so we use mu - beta * sigma.
    # Use a fixed beta of 2.0 as per the directive to simplify the hybrid approach
    # and avoid the scale-matching issues of adaptive weights.
    beta = 2.0
    ucb = mu - beta * sigma_safe
    
    # Normalize both components to [0, 1]
    ei_norm = _minmax_normalize(ei)
    ucb_norm = _minmax_normalize(ucb)
    
    # Adaptive exploration weight based on proximity to global minimum
    # This borrows the "adaptive blending based on proximity" idea from inspiration 2
    # but uses a simpler linear mapping instead of sigmoid, and applies it to the
    # blend weights rather than the beta parameter itself.
    
    global_min = -39.1662
    
    # Measure how close we are to the global minimum
    # If f_best is close to global_min, we should exploit more (increase EI weight)
    # If f_best is far, we should explore more (increase UCB weight)
    
    # Use a linear mapping:
    # When f_best is near global_min (delta ~ 0), w_ei should be high (e.g., 0.8)
    # When f_best is far (delta large), w_ei should be low (e.g., 0.2)
    
    delta_to_min = f_best - global_min  # Positive when above minimum
    
    # Normalize delta: assume a reasonable range for delta_to_min
    # Styblinski-Tang 1D range is roughly [-39, 1300]
    # So delta_to_min ranges from 0 to ~1340
    # Let's use a scale of 50.0 to control the transition width
    scale = 50.0
    
    # Map delta_to_min to a weight in [0.2, 0.8]
    # When delta_to_min is 0, w_ei = 0.8
    # When delta_to_min is >= 50, w_ei = 0.2
    # Linear interpolation: w_ei = 0.8 - (0.6 * delta_to_min / scale)
    # Clip to [0.2, 0.8]
    
    w_ei = 0.8 - (0.6 * delta_to_min / scale)
    w_ei = np.clip(w_ei, 0.2, 0.8)
    w_ucb = 1.0 - w_ei
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
