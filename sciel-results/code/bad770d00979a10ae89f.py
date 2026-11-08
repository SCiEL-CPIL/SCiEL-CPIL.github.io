"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Hybrid best-fit/worst-fit to minimize excess over L1 lower bound.
    
    Key insights:
    1. Pure best-fit can cause fragmentation (many small gaps that can't be filled).
    2. Pure worst-fit wastes capacity by spreading items too thin.
    3. A hybrid that strongly prefers tight fits but slightly favors preserving
       large contiguous space for future large items works well for skewed
       distributions like Weibull.
    
    Scoring:
    - Primary: -waste (best-fit component, minimize remaining space)
    - Secondary: +alpha * bins[i] (worst-fit component, preserve capacity)
    
    The alpha weight balances these two objectives. A small alpha (0.05-0.15)
    slightly favors worst-fit while keeping best-fit as the dominant strategy.
    
    Additionally, we apply a nonlinear penalty for very small gaps (waste < item),
    since such gaps are likely unusable for future items, effectively making
    them "dead" space.
    """
    remaining = bins - item
    valid = remaining >= 0
    
    scores = np.full(len(bins), -np.inf)
    if np.any(valid):
        waste = remaining[valid]
        bin_rem = bins[valid]
        
        # Base score: best-fit (minimize waste)
        base_score = -waste
        
        # Worst-fit component: slightly prefer bins with more remaining capacity
        # This helps preserve large contiguous space for future large items
        worst_fit_bonus = 0.1 * bin_rem
        
        # Nonlinear penalty for very small gaps that are likely unusable
        # If waste is small relative to the item, it's probably dead space
        gap_ratio = waste / (item + 1e-9)
        # Penalize gaps that are less than ~50% of current item size
        # These are unlikely to be filled by future items (which are similar scale)
        small_gap_penalty = np.where(gap_ratio < 0.5, 0.5 * waste, 0.0)
        
        scores[valid] = base_score + worst_fit_bonus - small_gap_penalty
    
    return scores


# EVOLVE-BLOCK-END
