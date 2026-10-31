import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remainders
    remainders = bins - item
    
    # Valid bins are those with enough capacity
    valid = bins >= item
    
    # Initialize scores to -inf for invalid bins
    scores = np.full_like(bins, -np.inf)
    
    if not np.any(valid):
        return scores
        
    r = remainders[valid]
    
    # Adaptive weight:
    # Large items (close to capacity) -> Best Fit (minimize remainder)
    # Small items -> Neutral (any valid bin)
    
    # w = 1.0 means Best Fit (score = -r)
    # w = 0.5 means Neutral (score = 0)
    
    # Linear interpolation based on item size relative to capacity (100)
    # item in [0, 100]
    # w = 0.5 + 0.5 * (item / 100.0)
    
    # If item is very small, w is close to 0.5.
    # If item is 100, w is 1.0.
    
    # To be safer, maybe cap item at 100?
    # The problem states capacity 100.
    
    w = 0.5 + 0.5 * (item / 100.0)
    
    # Score formula:
    # Best Fit: -r
    # Neutral: 0
    # Linear blend: w * (-r) + (1-w) * 0 = -w * r
    
    # Wait, if w=0.5, score = -0.5 * r. This still prefers smaller r.
    # "Neutral" should mean score is independent of r.
    # So we need a blend between Best Fit (-r) and a constant (0).
    # Or blend between Best Fit (-r) and Worst Fit (r)?
    # The inspiration said "neutral (rather than Worst Fit)".
    # So target for small items is 0.
    
    # score = w * (-r) + (1-w) * 0 = -w * r
    # If w=0.5, score = -0.5 * r. This is still Best Fit, just weaker.
    # Is that "Neutral"?
    # If the evaluator uses `argmax`, and scores are proportional to -r, it picks the smallest r.
    # To be truly neutral, the score should not depend on r.
    
    # So we should blend between Best Fit (-r) and a constant score (e.g., 0).
    # Let alpha be the weight of Best Fit.
    # score = alpha * (-r) + (1-alpha) * 0
    # If alpha=1, Best Fit.
    # If alpha=0, Neutral.
    
    # Let's map item size to alpha.
    # Large item -> alpha=1.
    # Small item -> alpha=0.
    
    # alpha = item / 100.0
    # If item=0, alpha=0, score=0.
    # If item=100, alpha=1, score=-r.
    
    alpha = item / 100.0
    scores[valid] = -alpha * r
    
    return scores
