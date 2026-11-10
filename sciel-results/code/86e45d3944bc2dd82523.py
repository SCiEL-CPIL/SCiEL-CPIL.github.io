"""Improved heuristic for ant colony offline bin packing."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    cap = float(capacity)

    # Pairwise sum of item sizes
    pair = d[:, None] + d[None, :]

    # A pair is "feasible" if their sum <= capacity
    feasible = pair <= cap

    # Remaining space after placing both items
    remaining = cap - pair

    # For feasible pairs, reward pairs that use MORE of the capacity (smaller remaining)
    # Use inverse gap: 1 / (remaining + eps) to strongly reward near-full bins
    # For infeasible pairs, set to a tiny value
    eps = 1e-6
    gap = np.where(feasible, remaining + eps, 1.0)
    
    # Primary score: inverse of remaining space (rewards filling bins)
    scores = np.where(feasible, 1.0 / gap, 1e-9)
    
    # Add a small bonus for pairs where items are similar in size (helps with fragmentation)
    size_diff = np.abs(d[:, None] - d[None, :])
    similarity_bonus = 1.0 / (1.0 + size_diff / cap)
    
    # Combine: primary inverse-gap with a small similarity component
    scores = scores * (1.0 + 0.1 * similarity_bonus)
    
    # Ensure all values are positive and finite
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 1e-9)

    return scores
# EVOLVE-BLOCK-END
