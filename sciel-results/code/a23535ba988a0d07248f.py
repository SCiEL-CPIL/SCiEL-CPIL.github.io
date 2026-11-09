import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    valid = remaining >= 0
    
    if not np.any(valid):
        return scores
        
    r = remaining[valid]
    
    # Best Fit Base
    base_score = -r
    
    # Bump for "Slightly Loose" fits
    # Center at 0.1 * item, Sigma 0.05 * item
    center = 0.1 * item
    sigma = 0.05 * item + 1e-10
    bump = np.exp(-((r - center) ** 2) / (2.0 * sigma ** 2))
    
    # Weight the bump
    # If weight is too high, we deviate too much from Best Fit.
    # Let's try a moderate weight.
    weight = 0.5
    
    combined = base_score + weight * bump
    
    scores[valid] = combined
    return scores
