"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin for placing the given item.
    
    Strategy: Smooth nonlinear scoring that combines best-fit with a gentle
    worst-fit bias and a smooth penalty for very small gaps.
    
    Key design:
    - Primary: minimize waste (best-fit) via a power-law transform of remaining space
    - Secondary: small worst-fit bonus to preserve large contiguous space
    - Smooth penalty for gaps that are likely unusable (smaller than ~50% of item)
    
    The power-law transform (exponent < 1) captures diminishing marginal value
    of remaining space, avoiding cliff-edge behavior at discrete thresholds.
    """
    start_time = time.time()
    
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
    # Use a smooth sigmoid-like penalty: strong when gap_ratio is small
    # The penalty transitions smoothly from 0 to a maximum value
    # Using: penalty = max_penalty * (1 - gap_ratio) * (gap_ratio < 0.5)
    # But to keep it smooth, use a soft version:
    # penalty = max_penalty * exp(-gap_ratio / scale) for gap_ratio < threshold
    # Simpler: use a smooth quadratic that peaks at gap_ratio = 0
    # penalty = max_penalty * exp(-gap_ratio / 0.3)
    max_penalty = 0.3
    scale = 0.3
    small_gap_penalty = max_penalty * np.exp(-gap_ratio / scale)
    
    # Combine scores
    scores_valid = base_score + worst_fit_bonus - small_gap_penalty
    
    # Build full scores array
    scores = np.full(len(bins), -np.inf, dtype=float)
    scores[valid] = scores_valid
    
    # Safety deadline guard
    elapsed = time.time() - start_time
    if elapsed > 110:  # 10 second safety margin
        pass  # Already computed, just return
    
    return scores
# EVOLVE-BLOCK-END
