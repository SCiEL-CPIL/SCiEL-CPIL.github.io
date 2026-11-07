"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    W = np.asarray(demands, dtype=np.float64)
    n = len(W)
    
    # Compute all pair sums
    pair_sums = W[:, None] + W[None, :]
    fit_mask = pair_sums <= capacity
    
    # Gap: how much space is left in the bin after placing i and j
    gap = capacity - pair_sums
    
    # Fill ratio: how full the bin is after placing i and j
    fill_ratio = pair_sums / capacity  # in [0, 1] for fitting pairs
    
    # Term 1: Inverse gap with softening to avoid extreme values
    # Use 1/(gap + c) where c is a softening constant
    # This rewards tight fits but doesn't blow up as gap -> 0
    c_soft = 10.0  # softening constant
    inv_gap_soft = np.where(fit_mask, 1.0 / (gap + c_soft), 0.0)
    
    # Term 2: Fill ratio squared - rewards high fill ratios
    # This captures the "tight packing" preference smoothly
    fill_sq = np.where(fit_mask, fill_ratio ** 2, 0.0)
    
    # Term 3: Size score - prefer larger items (more constraining)
    size_score = (W[:, None] + W[None, :]) / capacity
    
    # Term 4: Sweet spot bonus - reward remaining capacity in [20, 60] range
    # This avoids both too-tight (can't fit more) and too-loose (wasted space)
    # Use a Gaussian-like bump centered at 40 with width 25
    remaining = gap
    sweet_center = 40.0
    sweet_width = 25.0
    sweet_bonus = np.where(fit_mask, np.exp(-((remaining - sweet_center) ** 2) / (2 * sweet_width ** 2)), 0.0)
    
    # Combine with tuned weights
    alpha = 1.0    # inverse gap
    beta = 0.8     # fill ratio squared
    gamma = 0.2    # size score
    delta = 0.5    # sweet spot bonus
    
    H = alpha * inv_gap_soft + beta * fill_sq + gamma * size_score + delta * sweet_bonus
    
    # Set non-fitting pairs to very small value
    H = np.where(fit_mask, H, 1e-9)
    
    # Ensure no negative or zero values for fitting pairs
    H = np.maximum(H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
