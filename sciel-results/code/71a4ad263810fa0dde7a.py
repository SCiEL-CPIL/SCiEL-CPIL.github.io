"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track running statistics of item sizes to estimate typical future item size
_item_sum = 0.0
_item_count = 0
_item_sq_sum = 0.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin for placing the given item.
    
    Strategy: Smooth nonlinear scoring that combines best-fit with a gentle
    worst-fit bias, a smooth penalty for very small gaps, and a Next-Item Fit
    proxy that aligns remaining capacity with typical future item sizes.
    
    Key design:
    - Primary: minimize waste (best-fit) via a power-law transform of remaining space
    - Secondary: small worst-fit bonus to preserve large contiguous space
    - Smooth penalty for gaps that are likely unusable (smaller than ~50% of item)
    - Next-Item Fit: reward bins whose post-placement remaining capacity closely
      matches the median item size, reducing fragmentation for typical future items
    
    The power-law transform (exponent < 1) captures diminishing marginal value
    of remaining space, avoiding cliff-edge behavior at discrete thresholds.
    """
    global _item_sum, _item_count, _item_sq_sum
    
    start_time = time.time()
    
    # Update running statistics
    _item_sum += item
    _item_count += 1
    _item_sq_sum += item * item
    
    # Filter bins with enough capacity
    valid = bins >= item
    if not np.any(valid):
        return np.full_like(bins, -np.inf, dtype=float)
    
    # Calculate remaining capacity after placing item
    remaining = bins - item
    
    # Only compute scores for valid bins
    waste = remaining[valid]
    bin_rem = bins[valid]
    
    # Base score: power-law transform of waste to capture diminishing marginal value
    # Using exponent 0.8 means small wastes are penalized more strongly,
    # but the function is smooth (no discrete thresholds)
    # We want HIGH score for LOW waste, so use negative power-law
    eps = 1e-9
    waste_safe = np.maximum(waste, eps)
    base_score = -np.power(waste_safe, 0.8)
    
    # Worst-fit component: slightly prefer bins with more remaining capacity
    # This helps preserve large contiguous space for future large items
    worst_fit_bonus = 0.05 * bin_rem
    
    # Smooth penalty for very small gaps that are likely unusable
    # If waste is small relative to the item, it's probably dead space
    gap_ratio = waste / (item + eps)
    max_penalty = 0.3
    scale = 0.3
    small_gap_penalty = max_penalty * np.exp(-gap_ratio / scale)
    
    # Next-Item Fit proxy: score bins based on how well their remaining capacity
    # matches the typical (median) item size. This reduces fragmentation by
    # keeping bins at a "ready" state for the next typical item.
    # 
    # For Weibull distributions, the median is close to the mean for shape ~1.
    # We use the running mean as a smooth proxy for the median.
    if _item_count > 0:
        mean_item = _item_sum / _item_count
        # Use a smoothed estimate: blend running mean with current item
        # to avoid instability early in the sequence
        if _item_count < 50:
            typical_item = (mean_item + item) / 2.0
        else:
            typical_item = mean_item
        
        # Score: negative squared difference between remaining capacity and typical item size
        # This is a smooth Gaussian-like kernel centered at typical_item
        # Higher score when remaining ≈ typical_item
        diff = waste - typical_item
        # Use a scale that's proportional to typical_item for robustness
        nif_scale = max(typical_item * 0.5, 1.0)
        next_item_fit = -np.power(diff / nif_scale, 2)
        # Scale the contribution
        nif_weight = 0.4
        next_item_fit_score = nif_weight * next_item_fit
    else:
        next_item_fit_score = 0.0
    
    # Combine scores
    scores_valid = base_score + worst_fit_bonus - small_gap_penalty + next_item_fit_score
    
    # Build full scores array
    scores = np.full(len(bins), -np.inf, dtype=float)
    scores[valid] = scores_valid
    
    # Safety deadline guard
    elapsed = time.time() - start_time
    if elapsed > 110:  # 10 second safety margin
        pass  # Already computed, just return
    
    return scores
# EVOLVE-BLOCK-END
