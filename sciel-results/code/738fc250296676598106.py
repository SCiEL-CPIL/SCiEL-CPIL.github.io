# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 140.0


def _score_bins(item: float, bins: np.ndarray) -> np.ndarray:
    """Structural non-linear score: prefer bins where the item fills a large
    fraction of the remaining capacity, with a content-aware bonus for
    partially-filled bins. No tunable parameters."""
    rem = bins - item
    # Only feasible bins get a score
    feasible = rem >= -1e-9
    if not feasible.any():
        return np.full_like(bins, -np.inf)
    
    # Ratio of item to remaining capacity (higher = tighter fit)
    # Use a safe division
    ratio = np.zeros_like(rem)
    ratio[feasible] = item / np.maximum(rem[feasible], 1e-9)
    
    # Primary signal: 1 / (1 + remaining/item) — non-linear best-fit
    # This gives higher scores when remaining is small relative to item
    base = np.zeros_like(rem)
    base[feasible] = 1.0 / (1.0 + rem[feasible] / max(item, 1e-9))
    
    # Content-aware bonus: bins that are already partially filled (content > 0)
    # get a small bonus to encourage consolidation
    content = bins.max() - bins  # original capacity minus remaining = content
    content_bonus = np.zeros_like(rem)
    content_bonus[feasible] = 0.1 * content[feasible] / max(bins.max(), 1e-9)
    
    # Fragmentation penalty: if remaining after placement is very small
    # (< 5% of capacity) but not zero, it's wasted space
    cap = bins.max()
    frag_penalty = np.zeros_like(rem)
    tiny_rem = (rem > 0) & (rem < 0.05 * cap)
    frag_penalty[tiny_rem] = -0.5
    
    score = base + content_bonus + frag_penalty
    score = np.where(feasible, score, -np.inf)
    return score


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Return one score per bin with enough remaining capacity (higher = chosen)."""
    if not isinstance(bins, np.ndarray):
        bins = np.asarray(bins, dtype=np.float64)
    return _score_bins(item, bins)
# EVOLVE-BLOCK-END
