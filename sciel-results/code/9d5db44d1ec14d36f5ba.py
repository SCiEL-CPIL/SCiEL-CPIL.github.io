import numpy as np
from scipy.stats import norm

# Global state to estimate t
_call_count = 0
_prev_f_best = None

def acquisition(mu, sigma, f_best):
    global _call_count, _prev_f_best
    
    # Detect new seed: if f_best increases, it's a new run
    if _prev_f_best is not None and f_best > _prev_f_best:
        _call_count = 0
    _prev_f_best = f_best
    _call_count += 1
    
    t = _call_count
    
    # Calculate beta_t = sqrt(2t log(1+t))
    # To avoid log(1+0) issues if t=0, but t>=1 here.
    beta = np.sqrt(2 * t * np.log(1 + t))
    
    # Clip beta to a reasonable range to prevent extreme exploration in later steps
    # For t=25, beta = sqrt(50 * log(26)) = sqrt(50 * 3.25) = sqrt(162.5) ~ 12.7
    # This is quite high. Standard UCB often uses smaller betas.
    # However, the directive specifies this formula.
    # Maybe scale it down? The directive says "using ... beta_t = ...". I should follow it.
    # But if it performs poorly, I might need to adjust.
    # Let's stick to the formula but maybe cap it?
    # No, let's trust the formula first.
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB term
    ucb = mu + beta * sigma_safe
    
    # Expected Improvement term
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
        
    # Combine
    # The directive says "combine it additively with a normalized Expected Improvement term"
    # Weight for EI? The previous code used 0.5.
    # Let's use 0.5 again, or maybe 1.0?
    # UCB is in objective units. EI_norm is in [0,1].
    # If UCB is large, EI_norm is negligible.
    # If UCB is small, EI_norm adds a constant bias.
    # This might favor points with low uncertainty if UCB is flat.
    # Let's try weight 0.5.
    
    score = ucb + 0.5 * ei_norm
    
    return score
