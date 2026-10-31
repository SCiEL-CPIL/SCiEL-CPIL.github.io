"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: keep a module-level start time to enforce the 120s limit.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 5s safety margin before 120s kill


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Returns an array of scores (higher is better for selection).
    
    Strategy: Best-Fit Decreasing heuristic variant.
    - Prefer bins that minimize the remaining space after placing the item.
    - Tie-breaking: prefer bins with larger absolute remaining capacity (to avoid fragmentation of small bins).
    - This is a simple, fast heuristic that often performs well on bin packing.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return best-fit scores quickly
        remaining = bins - item
        mask = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(mask):
            scores[mask] = -remaining[mask]
        return scores

    remaining = bins - item
    mask = remaining >= 0

    scores = np.full(len(bins), -np.inf)
    if not np.any(mask):
        return scores

    # Primary: minimize remaining space (best-fit)
    # Secondary: among ties, prefer larger original bin capacity (to preserve small bins for small items)
    # Actually, standard best-fit is just -remaining. Let's stick with that for simplicity and speed.
    # But we can add a small tie-breaker to prefer bins that are "more full" already? No, best-fit already does that.
    # Let's try a slight variation: Best-Fit with a penalty for creating a new bin if possible?
    # The interface doesn't tell us which bins are "new". It just gives current capacities.
    # So we assume bins array contains the remaining capacity of all open bins.
    
    # Let's use a smooth score: -remaining + small * (bins) to break ties towards larger bins?
    # Or just -remaining is robust.
    
    # Let's try: score = -remaining
    # This is the seed. Let's see if we can improve.
    # Common improvement: Worst-Fit Decreasing (WFD) sometimes works better for random sizes?
    # Or First-Fit Decreasing (FFD) equivalent?
    # In online setting, FFD is not directly applicable without knowing order.
    # Best-Fit (BF) is a strong baseline.
    
    # Let's try a hybrid: if remaining is very small, prefer it. If remaining is large, maybe prefer it to avoid fragmentation?
    # Actually, "Next-Fit" is bad. "First-Fit" is okay. "Best-Fit" is usually better than "Worst-Fit" for uniform distributions.
    # For Weibull (skewed), Best-Fit is generally strong.
    
    # Let's stick with Best-Fit but ensure numerical stability.
    scores[mask] = -remaining[mask]
    
    return scores
# EVOLVE-BLOCK-END
