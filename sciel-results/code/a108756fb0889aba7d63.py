import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the given item.
    The item goes to the highest-scoring bin.
    Lower total bins used is better (lower excess over L1 lower bound).
    """
    # Explicit deadline guard: track elapsed time
    # For a single item placement, this is negligible, but we keep the structure.
    # In a full evaluation loop, this would be more critical.
    
    remaining = bins - item
    valid_mask = remaining >= 0
    
    if not np.any(valid_mask):
        return np.full_like(bins, -np.inf)
    
    scores = np.full_like(bins, -np.inf)
    
    # Design: Hybrid Best Fit / Worst Fit
    # Based on the observed improvement from inverting the logic:
    # - Small items: Best Fit (w=0) -> maximize score for smallest remaining (tightest fit)
    #   Wait, let's re-evaluate the scoring formula.
    #   scores = (2*w - 1) * remaining
    #   If w=0 (Best Fit): scores = -remaining. Highest score is the one with the SMALLEST remaining. This is Best Fit Decreasing style.
    #   If w=1 (Worst Fit): scores = +remaining. Highest score is the one with the LARGEST remaining. This is Worst Fit Decreasing style.
    
    # The previous program used:
    # low=40, high=85
    # w=0 for item <= 40 (Best Fit)
    # w=1 for item >= 85 (Worst Fit)
    
    # Let's refine the transition.
    # Weibull distribution of items. Capacity 100.
    # Typical Weibull parameters might generate many small items and some large ones.
    # "Consolidating" small items (Best Fit) is generally good.
    # For large items, Worst Fit is often better to avoid fragmenting large holes,
    # OR Best Fit is better to fill holes tightly.
    # The cell finding said: "Inverting the hybrid heuristic to use Worst Fit for large items ... and Best Fit for small items ... addresses the likely cause".
    # So: Small -> Best Fit (w=0). Large -> Worst Fit (w=1).
    # This matches the current logic structure if we interpret w=0 as Best Fit and w=1 as Worst Fit.
    
    # Let's tweak the thresholds and the smoothness.
    # Maybe the transition is too sharp or the thresholds are slightly off.
    # Let's try a slightly wider transition and see.
    # Or maybe pure Best Fit is better for most, and only very large items need Worst Fit?
    
    low = 50.0
    high = 90.0
    
    if item <= low:
        w = 0.0
    elif item >= high:
        w = 1.0
    else:
        t = (item - low) / (high - low)
        # Smoothstep
        w = t * t * (3.0 - 2.0 * t)
        
    # Blend
    # We want to select the bin with the highest score.
    # For w=0: score = -remaining. Max score = min remaining. (Best Fit)
    # For w=1: score = remaining. Max score = max remaining. (Worst Fit)
    
    scores[valid_mask] = (2.0 * w - 1.0) * remaining[valid_mask]
    
    return scores
