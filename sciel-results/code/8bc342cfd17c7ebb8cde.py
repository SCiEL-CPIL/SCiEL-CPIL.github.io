"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for adaptive parameters
_start_time = time.time()
_deadline = 145.0  # seconds (safety margin before 150s kill)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score = better choice.
    
    Uses a refined best-fit heuristic with gap-quality penalty:
    - Primary: minimize remaining space (best-fit)
    - Secondary: penalize gaps that are "awkward" (small but not small enough
      to be filled by typical future items)
    
    The score combines:
    1. -remaining (best-fit: prefer bins where item fits tightly)
    2. A smooth penalty for gap ratios in the "danger zone" (0.05-0.3 of item)
       These gaps are too small to be useful for future items but too large
       to ignore.
    
    For Weibull-distributed items (typically right-skewed), this helps avoid
    creating unusable small gaps while still preferring tight fits.
    """
    remaining = bins - item
    valid = remaining >= 0
    
    if not np.any(valid):
        return np.full(len(bins), -np.inf)
    
    # Base score: negative remaining (best-fit)
    # Scale by item to make scores comparable across different item sizes
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_gap = np.where(valid, remaining / max(item, 1e-10), 0.0)
    
    # Best-fit component: prefer smaller gaps
    best_fit_score = -rel_gap
    
    # Gap-quality penalty: penalize gaps in the "danger zone"
    # Gaps between ~0.05 and ~0.3 of item size are problematic
    # They're too small for most future items but create waste
    # Use a smooth bump function centered at rel_gap=0.15, width=0.1
    center = 0.15
    width = 0.12
    # Smooth penalty: Gaussian-like bump that peaks in the danger zone
    penalty = np.exp(-((rel_gap - center) ** 2) / (2 * width ** 2))
    # Only apply penalty to valid bins
    penalty = np.where(valid, penalty, 0.0)
    
    # Weight the penalty relative to best-fit
    # For small items, the danger zone is more critical
    # Scale penalty weight inversely with item size (capped)
    item_scale = min(1.0, 1.0 / max(item, 1.0))
    penalty_weight = 0.8 * item_scale
    
    score = best_fit_score - penalty_weight * penalty
    score = np.where(valid, score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
