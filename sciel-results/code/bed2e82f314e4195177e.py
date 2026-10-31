"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0
    
    # Fluid-flow scoring: prioritize bins where item fills a large fraction
    # of remaining capacity (relative fit), with smooth gap penalty.
    
    # Relative fill ratio: item / (item + remaining) = item / bins
    # Higher ratio means item is a larger fraction of the bin's capacity
    # Use bins directly to avoid division by zero issues
    fill_ratio = np.where(bins > 0, item / bins, 0.0)
    
    # Base score: maximize fill ratio (fluid flow - minimize relative waste)
    score = fill_ratio.copy()
    
    # Smooth penalty for creating very small unusable gaps
    # Gaps below a threshold relative to item size are wasteful
    gap_threshold = max(1.0, 0.02 * item)
    diff = gap_threshold - remaining
    gap_penalty = np.where(diff > 0, 0.5 * diff * diff / (gap_threshold * gap_threshold), 0.0)
    
    # Also penalize very large gaps (waste of bin capacity)
    # But much more gently than small gaps
    large_gap_threshold = 0.3 * item
    large_diff = remaining - large_gap_threshold
    large_penalty = np.where(large_diff > 0, 0.1 * large_diff / (large_gap_threshold + 1.0), 0.0)
    
    score = score - gap_penalty - large_penalty
    
    # Invalidate bins that can't fit the item
    score = np.where(valid, score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
