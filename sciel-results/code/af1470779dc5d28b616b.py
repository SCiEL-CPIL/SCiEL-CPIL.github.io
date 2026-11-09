# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 140.0


def _score_bins(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Harmonic mean-based scoring: score each feasible bin by the harmonic mean
    of its remaining capacity and the item size. This prioritizes placements
    that minimize relative waste compared to the item's demand.
    
    The harmonic mean H(a, b) = 2ab/(a+b) is dominated by the smaller of a and b,
    so it strongly penalizes cases where remaining capacity is much larger than
    the item (wasteful) or where the item is much larger than remaining (infeasible).
    For feasible bins, we want high harmonic mean of (remaining, item), which
    corresponds to tight fits where remaining ≈ item.
    """
    rem = bins - item
    # Only feasible bins get a score
    feasible = rem >= -1e-9
    if not feasible.any():
        return np.full_like(bins, -np.inf)
    
    # Harmonic mean of remaining capacity and item size
    # H(rem, item) = 2 * rem * item / (rem + item)
    # This is maximized when rem == item, and decreases as they diverge
    # We use this as the primary score
    rem_safe = np.maximum(rem, 1e-9)
    item_safe = max(item, 1e-9)
    
    # Harmonic mean: high when rem ≈ item, low when rem >> item or rem << item
    harmonic = 2.0 * rem_safe * item_safe / (rem_safe + item_safe)
    
    # Normalize to [0, 1] for consistent scaling
    cap = bins.max()
    harmonic_norm = harmonic / max(cap, 1e-9)
    
    # Primary score: positive harmonic mean (higher is better)
    score = np.full_like(bins, -np.inf)
    score[feasible] = harmonic_norm[feasible]
    
    # Secondary: small bonus for bins with existing content (consolidation)
    # This helps prefer partially-filled bins over empty ones when harmonic is similar
    content = cap - bins
    content_bonus = np.zeros_like(bins)
    content_bonus[feasible] = 0.05 * content[feasible] / max(cap, 1e-9)
    
    score[feasible] += content_bonus[feasible]
    
    return score


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Return one score per bin with enough remaining capacity (higher = chosen)."""
    if not isinstance(bins, np.ndarray):
        bins = np.asarray(bins, dtype=np.float64)
    
    # Deadline guard: if we're running out of time, use simple best-fit
    if time.time() > _DEADLINE:
        remaining = bins - item
        scores = np.full_like(bins, -np.inf)
        valid = remaining >= -1e-9
        scores[valid] = -remaining[valid]
        return scores
    
    return _score_bins(item, bins)
# EVOLVE-BLOCK-END
