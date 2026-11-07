"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    pair = demands[:, None] + demands[None, :]
    valid = pair <= capacity
    # Fill ratio: how much of the bin is used by this pair
    fill_ratio = pair / capacity
    # Cubic power of fill ratio: sharply rewards near-full pairs
    # (fill_ratio^3) penalizes low-fill pairs more than quadratic
    cubic_fill = fill_ratio ** 3
    # Gaussian damping: penalize pairs with fill_ratio < 0.4
    # exp(-((fill_ratio - 0.4) / 0.15)^2) for fill_ratio < 0.4, else 1.0
    sigma = 0.15
    center = 0.4
    damp_raw = np.exp(-((fill_ratio - center) / sigma) ** 2)
    # Only apply damping where fill_ratio < center
    damp = np.where(fill_ratio < center, damp_raw, 1.0)
    # Combined score: cubic_fill * damp
    score = np.where(valid, cubic_fill * damp, 0.0)
    # Ensure all entries are positive (minimum 1e-9)
    score = np.maximum(score, 1e-9)
    return score
# EVOLVE-BLOCK-END
