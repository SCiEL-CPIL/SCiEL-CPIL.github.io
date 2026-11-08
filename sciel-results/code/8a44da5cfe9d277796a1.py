"""Rewritten program for online bin packing; stronger heuristic than best-fit."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We can use a simple, fast heuristic that combines best-fit with a slight
# preference for bins that are "medium" full, to avoid opening new bins too early
# while also not over-packing bins that might become unusable.
#
# Idea: Score = -remaining_capacity + small_bonus * (1 - |utilization - 0.5|)
# But since we only see bins (remaining capacities) and the item, and we want
# to minimize excess over L1 lower bound, we should prefer bins that minimize
# waste. Best-fit is known to be strong. Let's try a variant that slightly
# prefers bins that are closer to half-full to keep options open, but primarily
# uses best-fit.
#
# Actually, let's think about what minimizes excess over L1 lower bound.
# The L1 lower bound is sum(items) / capacity. Excess = (num_bins - LB) / LB.
# To minimize num_bins, we want to pack tightly. Best-fit does this well.
#
# Let's try: score = -(bins - item) + 0.01 * (1 - abs(bins/bins_max - 0.5))
# But bins_max is not known. We can use the max of bins as a proxy.
#
# Simpler: Let's try a weighted combination of best-fit and a "balance" term.
# balance = 1 - abs(remaining/bins_max - 0.5)  # prefer bins that are ~half full
# But this requires bins_max. Let's compute it.
#
# Actually, let's just use a slightly modified best-fit that penalizes bins
# that are too close to empty or too close to full.
#
# Let's try: score = -(bins - item) + 0.005 * (bins / max_bin) * (1 - bins / max_bin)
# This adds a small bonus for bins that are around half-full.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Filter bins that can fit the item
    mask = bins >= item
    if not np.any(mask):
        return np.zeros_like(bins)
    
    remaining = bins[mask]
    
    # Best-fit score: negative remaining capacity after placing item
    best_fit_score = -(remaining - item)
    
    # Balance term: prefer bins that are around half-full
    # Use the maximum bin capacity as reference
    max_bin = np.max(bins) if len(bins) > 0 else 1.0
    if max_bin > 0:
        utilization = remaining / max_bin
        balance = 0.01 * utilization * (1 - utilization)
    else:
        balance = 0.0
    
    # Combined score
    scores = best_fit_score + balance
    
    # Create full score array with -inf for infeasible bins
    result = np.full_like(bins, -np.inf, dtype=float)
    result[mask] = scores
    
    return result
# EVOLVE-BLOCK-END
