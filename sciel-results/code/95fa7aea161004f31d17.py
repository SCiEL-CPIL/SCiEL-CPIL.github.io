import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
DEADLINE = 148.0  # 150s limit, 2s safety margin
SAFETY_MARGIN = 2.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline check
    if time.time() > DEADLINE - SAFETY_MARGIN:
        # Fallback to simple Best Fit
        feasible = bins >= item
        scores = np.full_like(bins, -np.inf, dtype=float)
        if np.any(feasible):
            scores[feasible] = -(bins[feasible] - item)
        return scores

    C = 100.0
    n_bins = len(bins)
    
    # Identify feasible bins
    feasible = bins >= item
    if not np.any(feasible):
        return np.full_like(bins, -np.inf, dtype=float)
    
    scores = np.full_like(bins, -np.inf, dtype=float)
    b = bins[feasible]
    
    # Strategy: Golden Ratio Attractor with Adaptive Weighting
    # 
    # The current program (Best Fit) scores 1.0000.
    # The inspiration suggests using a "golden ratio" target for the remainder.
    # The golden ratio phi = (1 + sqrt(5)) / 2 ≈ 1.618.
    # In bin packing, a remainder of roughly 1/phi * C or similar might be optimal?
    # Actually, the prompt says "negative Euclidean distance between the resulting remainder and a 'golden ratio' target (approx. 0.618 * capacity)".
    # Let's use target = 0.618 * C = 61.8.
    # 
    # However, pure Golden Fit might be worse than Best Fit for small items.
    # Best Fit minimizes waste immediately.
    # Golden Fit aims for a specific remainder.
    # 
    # Let's combine them:
    # We want to minimize the distance to the target remainder, but also penalize
    # creating very small remainders (dead space) if we are not close to the target?
    # 
    # Actually, let's look at the "REFINEMENT" directive:
    # "Model the bin state as a 1D spatial coordinate where the score is the negative 
    # Euclidean distance between the resulting remainder and a 'golden ratio' target"
    # 
    # Let's implement this directly but blend it with Best Fit based on the item size.
    # If the item is small, Best Fit is usually better because it fills bins tightly.
    # If the item is large, aiming for a specific remainder might help future packing.
    # 
    # Let's try a hybrid score:
    # score = -alpha * (remainder - item)  # Best Fit component (minimize remainder)
    #       - beta * (remainder - target)^2 # Golden Fit component (target specific remainder)
    # 
    # Wait, the score is maximized. So we want to maximize:
    # score = -alpha * (b - item) - beta * (b - item - target)^2
    # 
    # Let's normalize.
    # target = 0.618 * C
    # 
    # If we use pure Best Fit (beta=0), we get the seed.
    # If we use pure Golden Fit (alpha=0), we get the inspiration.
    # 
    # Let's try a dynamic alpha/beta.
    # If the bin is already close to the target, prefer Best Fit (tighten it).
    # If the bin is far from the target, prefer Golden Fit (move it towards target).
    # 
    # Actually, a simpler approach:
    # The "Golden Ratio" idea in bin packing often relates to the fact that if you have
    # a remainder R, you can fit another item of size R/phi. 
    # 
    # Let's stick to the prompt's specific instruction: 
    # "score is the negative Euclidean distance between the resulting remainder and a 'golden ratio' target"
    # 
    # resulting remainder = b - item
    # target = 0.618 * C
    # distance = | (b - item) - target |
    # score = -distance
    # 
    # But pure distance to a fixed target is risky.
    # Let's blend it with Best Fit.
    # 
    # Best Fit score: - (b - item)
    # Golden Fit score: - | (b - item) - target |
    # 
    # Let's interpolate.
    # If the best fit remainder is very small (< 5), Best Fit is great.
    # If the best fit remainder is large, maybe Golden Fit is better?
    # 
    # Let's try:
    # rem = b - item
    # target = 61.8
    # 
    # score_bf = -rem
    # score_gf = -np.abs(rem - target)
    # 
    # We want to use GF when rem is large, and BF when rem is small.
    # Let's use a smooth switch.
    # switch = sigmoid((rem - threshold) / temperature)
    # score = (1 - switch) * score_bf + switch * score_gf
    # 
    # Threshold: maybe 20?
    # Temperature: 5?
    # 
    # Let's refine the target.
    # The golden ratio conjugate is 1/phi = 0.618.
    # 
    # Let's try this hybrid.

    rem = b - item
    target = 0.618 * C
    
    # Best Fit component
    score_bf = -rem
    
    # Golden Fit component
    score_gf = -np.abs(rem - target)
    
    # Dynamic weighting
    # If rem is small, BF is better. If rem is large, GF might be better to preserve structure.
    # Let's use a sigmoid to blend.
    threshold = 20.0
    temp = 10.0
    switch = 1.0 / (1.0 + np.exp(-(rem - threshold) / temp))
    
    scores[feasible] = (1.0 - switch) * score_bf + switch * score_gf
    
    return scores
