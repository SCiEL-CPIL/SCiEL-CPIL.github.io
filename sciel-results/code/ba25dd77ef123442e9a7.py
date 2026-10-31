"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # safety margin before 150s kill
_START_TIME = time.time()

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard: if we're too close to the limit, return a safe default
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        # Fallback to simple Best Fit
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores

    # Determine which bins have enough remaining capacity
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)

    # Calculate the waste (remaining space) for each valid bin
    waste = bins - item
    
    # Structural Change: Adaptive Strategy based on Fragmentation Pressure
    # Instead of a fixed penalty (like -waste^2), we use a harmonic-like 
    # penalty that adapts to the "tightness" of the best fit available.
    
    # min_waste is the waste if we choose the Best Fit bin.
    min_waste = np.min(waste[valid])
    
    # If the best fit is very tight (min_waste is small relative to item),
    # we should strongly prefer Best Fit to avoid opening new bins unnecessarily.
    # If the best fit is loose (min_waste is large), it might be better to 
    # choose a bin that is not the absolute best fit but still reasonably good,
    # to keep the "tightest" bins available for future items that might fit 
    # only there (preserving fragmentation structure).
    
    # Let's define a "tightness" factor.
    # If min_waste is close to 0, tightness is high.
    # We use a smooth transition.
    
    # Strategy: 
    # Score = - (waste * alpha + 1/ (waste + epsilon) * beta)
    # This combines a linear penalty (prefer smaller waste) with a harmonic penalty
    # that explodes if waste is 0 (wait, if waste is 0, 1/waste is inf, so we'd avoid it? 
    # No, we want to AVOID large waste. 
    # Let's rethink.
    
    # Standard Best Fit: minimize waste. Score = -waste.
    # Standard Worst Fit: maximize waste. Score = waste.
    
    # The insight from the prompt: "Replacing the hard threshold with a smooth sigmoid... 
    # favors Best Fit for large items and becomes neutral... for small items"
    
    # Let's try a strategy that mimics "Next Fit Decreasing" intuition but online.
    # If the item is small relative to the bin capacity (implied by low waste in best fit),
    # we are in a "fragmentation risk" zone. We should perhaps avoid creating many tiny scraps.
    # Actually, Best Fit minimizes scraps. Worst Fit keeps bins full.
    
    # Let's try a "Weighted Best Fit" where the weight depends on the item size relative to capacity (100).
    # If item is small (< 20), use a softer penalty (more like First Fit or Random among valid) to explore?
    # No, exploration is bad for deterministic scoring.
    
    # New Idea: **Harmonic Best Fit**
    # Score = - (waste + 1/(waste + 1e-6))
    # This heavily penalizes waste near 0? No, 1/(waste+eps) is large when waste is small.
    # So - (waste + 1/(waste+eps)) is very negative for small waste. This would AVOID best fit.
    # That's Wrong Fit.
    
    # We want to PREFER small waste.
    # Score = - waste^k. Current is k=2.
    # What if k is adaptive?
    # If the best available waste is very small (tight pack), use k=1 (linear, standard BF).
    # If the best available waste is large (loose pack), use k=0.5 (sqrt, softer, more sensitive to differences)?
    
    # Let's look at the "Island" inspiration.
    # The prompt mentions a sigmoid transition.
    
    # Let's try: **Relative Waste Penalty**
    # Normalize waste by the item size?
    # waste_ratio = waste / item
    
    # If we use Score = - waste_ratio^2, it's similar to before.
    
    # Let's try a fundamentally different approach: **Lookahead-free but State-Aware**
    # We can't look ahead, but we can look at the *distribution* of valid bins.
    # If most valid bins are very large, we are in a "sparse" regime.
    # If most valid bins are tight, we are in a "dense" regime.
    
    # In sparse regime, Best Fit is good.
    # In dense regime, maybe we should preserve the tightest bins?
    # Actually, Best Fit is generally superior for online bin packing (L1 lower bound context).
    # The seed is Best Fit (score 1.0). The quadratic version is also 1.0.
    
    # To beat 1.0, we need to handle the specific Weibull distribution better.
    # Weibull items often have a mix of small and large.
    
    # Hypothesis: For small items, Best Fit creates many bins with tiny remaining space (fragmentation).
    # If we use a slightly "looser" fit for small items, we might keep bins with more remaining space,
    # which could accommodate future medium items better.
    
    # Strategy: **Item-Size-Dependent Exponent**
    # If item is small (< 20% of capacity), use a lower exponent (e.g., 0.5) to make the score less sensitive to waste differences, effectively acting more like "First Fit" or "Random Valid" but still preferring smaller waste.
    # If item is large, use a higher exponent (e.g., 2.0) to strictly enforce Best Fit.
    
    # Let's implement this.
    
    # Capacity is 100.
    # Threshold: 20.
    
    if item < 20:
        # Small item: Use sqrt(waste) penalty.
        # This makes the score difference between waste=1 and waste=2 smaller than linear.
        # It encourages picking a "good enough" bin rather than the absolute best,
        # potentially saving the absolute best for a slightly larger item.
        scores = -np.sqrt(waste)
    else:
        # Large item: Use quadratic penalty (strict Best Fit).
        scores = -waste**2
        
    # Mask invalid bins
    scores[~valid] = -np.inf
    
    return scores
# EVOLVE-BLOCK-END
