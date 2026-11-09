import numpy as np
from scipy.special import gamma
from scipy.optimize import brentq

# Global state for adaptive parameters
_last_items = []
_beta = 1.0
_lambda = 1.0
_last_update_count = 0

def _estimate_weibull(items):
    """Estimate Weibull shape and scale from a list of items."""
    if len(items) < 10:
        return 1.0, 1.0 # Default to Exponential
    
    items = np.array(items)
    mu = np.mean(items)
    var = np.var(items)
    if mu <= 0:
        return 1.0, 1.0
    
    cv2 = var / (mu * mu)
    
    # Function to solve for beta: Gamma(1+2/beta)/Gamma(1+1/beta)^2 - 1 - cv2 = 0
    def f(beta):
        g1 = gamma(1 + 1/beta)
        g2 = gamma(1 + 2/beta)
        return g2 / (g1 * g1) - 1 - cv2
    
    # Search for root in range [0.1, 10]
    try:
        beta = brentq(f, 0.1, 10)
    except ValueError:
        beta = 1.0
        
    lambda_scale = mu / gamma(1 + 1/beta)
    return beta, lambda_scale

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _last_items, _beta, _lambda, _last_update_count
    
    # Update parameters periodically
    _last_items.append(item)
    _last_update_count += 1
    
    if _last_update_count % 10 == 0:
        _beta, _lambda = _estimate_weibull(_last_items[-100:])
        # Keep buffer size manageable
        if len(_last_items) > 200:
            _last_items = _last_items[-100:]
            
    # Calculate remaining capacity after placement
    remaining = bins - item
    
    # Only consider bins that can fit the item
    # The prompt implies `bins` might contain all bins, or only valid ones.
    # To be safe, we mask invalid bins with a very low score.
    
    # Determine weight for Worst Fit vs Best Fit
    # If beta < 1, favor Worst Fit (keep space for large items)
    # If beta > 1, favor Best Fit (pack tight)
    
    # Map beta to weight w in [0, 1]
    # w=0 -> Best Fit (minimize remaining)
    # w=1 -> Worst Fit (maximize remaining)
    
    # Sigmoid mapping centered at beta=1
    # If beta=1, w=0.5 (Neutral? Or just Best Fit?)
    # Let's say beta=1 is standard Exponential, where Best Fit is good.
    # So let's bias towards Best Fit.
    
    # If beta < 1, w increases towards 1.
    # If beta > 1, w decreases towards 0.
    
    # Simple linear mapping clipped
    # beta=0.5 -> w=0.8
    # beta=1.0 -> w=0.2
    # beta=2.0 -> w=0.0
    
    if _beta < 1.0:
        w = 1.0 - (_beta - 0.5) * 2.0 # beta=0.5->1.0, beta=1.0->0.0
        w = np.clip(w, 0.0, 1.0)
    else:
        w = 0.0
        
    # Score calculation
    # Best Fit Score: -remaining
    # Worst Fit Score: remaining
    
    # Blend: w * remaining + (1-w) * (-remaining) = (2w - 1) * remaining
    
    coeff = 2 * w - 1
    
    scores = coeff * remaining
    
    # Mask invalid bins (bins < item)
    invalid = bins < item
    scores[invalid] = -np.inf
    
    return scores
