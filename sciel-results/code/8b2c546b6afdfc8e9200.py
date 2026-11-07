"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Compute an item-pair attractiveness matrix for ant colony optimization
    of offline bin packing.
    
    The heuristic guides ants to place item j after item i in a way that
    minimizes wasted space and leaves useful residuals for future items.
    """
    t0 = time.time()
    deadline = t0 + 320.0  # safety margin before 330s kill

    n = len(demands)
    d = np.array(demands, dtype=float)

    # Pair sums
    pair = d[:, None] + d[None, :]

    # Mask for pairs that fit
    fits = pair <= capacity

    # Initialize with very low attractiveness
    h = np.full((n, n), 1e-9, dtype=float)

    # Component 1: Fill ratio squared - rewards fuller packs quadratically
    fill_ratio = pair / capacity
    fill_score = np.where(fits, fill_ratio ** 2, 0.0)

    # Component 2: Inverse remaining space - rewards tight packing (sharpened)
    remaining = np.where(fits, capacity - pair, capacity)
    # Use a steeper curve: 1/(remaining+1)^2 amplifies the difference between
    # tight (remaining~5) and loose (remaining~50) packs
    inv_remaining = np.where(remaining > 1e-6, 1.0 / (remaining + 1.0) ** 2, 0.0)
    inv_remaining_score = np.where(fits, inv_remaining * capacity * capacity * 0.01, 0.0)

    # Component 3: Smooth residual quality - continuous signal replacing the
    # hard threshold at min_d. High when residual is very small (tight) or
    # close to a "useful" size (can fit another item).
    min_d = d.min()
    residual = np.where(fits, capacity - pair, 0.0)

    # Smooth bonus: peaks at residual=0 (perfect fill) and has a secondary
    # peak around residual=min_d (can fit one more smallest item)
    # Use a sum of two smooth bumps
    sigma_tight = min_d * 0.4 + 1.0
    tight_bump = np.exp(-residual ** 2 / (2 * sigma_tight ** 2))
    sigma_useful = min_d * 0.5 + 1.0
    useful_bump = np.exp(-(residual - min_d) ** 2 / (2 * sigma_useful ** 2))
    residual_bonus = np.where(fits, 0.35 * tight_bump + 0.15 * useful_bump, 0.0)

    # Combine components
    h = np.where(fits, fill_score + inv_remaining_score + residual_bonus, 1e-9)

    # Don't allow placing an item after itself
    np.fill_diagonal(h, 1e-9)

    # Ensure non-positive and non-finite values become 1e-9
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9

    # Check deadline before optional enhancement
    if time.time() > deadline - 5:
        return h

    # Vectorized triple-fit bonus: for each fitting pair (i,j), find the
    # largest item that fits into the residual and reward closeness
    sorted_d = np.sort(d)
    
    # For each pair (i,j), residual = capacity - d[i] - d[j]
    # Find the largest item <= residual using searchsorted
    r_flat = residual.ravel()
    fits_flat = fits.ravel()
    
    # For non-fitting pairs, residual is 0, searchsorted will give idx=-1
    # Clip to valid range
    idx = np.searchsorted(sorted_d, r_flat, side='right') - 1
    idx = np.clip(idx, 0, len(sorted_d) - 1)
    
    # Best fit item size for each pair
    best_fit_flat = sorted_d[idx]
    best_fit = best_fit_flat.reshape(n, n)
    
    # Tighter triple-fit bonus: reward only when the residual is well-utilized
    # by the next item. Use a sharper penalty for poor fit.
    r_safe = np.where(r_flat.reshape(n, n) > 1e-6, r_flat.reshape(n, n), 1.0)
    closeness = 1.0 - np.abs(residual - best_fit) / r_safe
    closeness = np.maximum(0.0, closeness)
    closeness_sq = closeness ** 2  # sharper: 0.5 -> 0.25, 0.9 -> 0.81
    
    # Only apply where fits and residual > 0 and i != j
    triple_mask = fits & (residual > 0) & ~np.eye(n, dtype=bool)
    triple_bonus = np.where(triple_mask, closeness_sq * 0.25, 0.0)
    
    h += triple_bonus
    
    # Component 4: Size balance bonus - reward pairs where both items are
    # reasonably sized (not one tiny + one huge). This promotes balanced bins.
    max_d_pair = np.maximum(d[:, None], d[None, :])
    diff = np.abs(d[:, None] - d[None, :])
    balance = 1.0 - diff / np.maximum(max_d_pair, 1e-6)
    balance_bonus = np.where(fits & ~np.eye(n, dtype=bool), balance * 0.15, 0.0)
    
    h += balance_bonus
    
    # Component 5: Multi-target residual bonus - reward residuals close to
    # common item sizes (modes of the size distribution). This encourages
    # packing toward sizes that actually appear in the dataset.
    targets = np.array([np.percentile(d, p) for p in [25, 50, 75]])
    target_bonus = np.zeros((n, n), dtype=float)
    for t_size in targets:
        residual_dist = np.abs(residual - t_size)
        sigma = t_size * 0.25 + 1.0
        bump = np.exp(-residual_dist ** 2 / (2 * sigma ** 2))
        target_bonus += bump
    target_bonus *= 0.08
    target_bonus = np.where(fits & ~np.eye(n, dtype=bool) & (residual > 0), target_bonus, 0.0)
    
    h += target_bonus

    # Final cleanup
    np.fill_diagonal(h, 1e-9)
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9

    return h
# EVOLVE-BLOCK-END
