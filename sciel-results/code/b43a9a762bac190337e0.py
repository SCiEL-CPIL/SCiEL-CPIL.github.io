"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 120.0  # seconds
_start_time = time.time()


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Safety: fall back to simple best fit
        return -(bins - item)
    
    remaining = bins - item
    feasible = remaining >= 0
    
    # Fill-ratio based scoring: prefer bins where the item fills a larger
    # fraction of the bin's remaining capacity. This reduces fragmentation
    # compared to pure best-fit by avoiding placing small items in large
    # nearly-empty bins when a tighter fit exists.
    # score = -item / bins for feasible bins (higher fill → higher score)
    # For infeasible bins, assign a very low score.
    safe_bins = np.where(feasible & (bins > 0), bins, 1.0)
    fill_ratio = item / safe_bins
    scores = np.where(feasible, fill_ratio, -np.inf)
    
    # Add a small tie-breaker: among equal fill ratios, prefer less remaining
    # capacity (classic best-fit tie-break)
    tie_break = -remaining / 1000.0
    scores = scores + tie_break
    
    return scores
# EVOLVE-BLOCK-END
