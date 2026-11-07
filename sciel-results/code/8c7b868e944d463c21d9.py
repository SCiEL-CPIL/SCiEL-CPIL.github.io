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
    deadline = t0 + 310.0  # safety margin before 330s kill

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

    # Component 2: Inverse remaining space - rewards tight packing
    remaining = np.where(fits, capacity - pair, capacity)
    inv_remaining = np.where(remaining > 1e-6, 1.0 / (remaining + 1.0), 0.0)
    inv_remaining_score = np.where(fits, inv_remaining * capacity * 0.5, 0.0)

    # Component 3: Residual quality - check if leftover can fit another item
    min_d = d.min()
    residual = np.where(fits, capacity - pair, 0.0)

    # Useful residual: either can fit another item (residual >= min_d)
    # or is very tight (residual < min_d * 0.3)
    useful_residual = np.where(fits,
                                (residual >= min_d) | (residual < min_d * 0.3),
                                False)
    residual_bonus = np.where(useful_residual, 0.3, 0.0)

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
    # residual is an (n,n) array; we need to vectorize searchsorted
    # searchsorted works on 1D arrays, so we flatten, search, then reshape
    
    r_flat = residual.ravel()
    # Only process fitting pairs
    fits_flat = fits.ravel()
    
    # For non-fitting pairs, residual is 0, searchsorted will give idx=-1
    # Clip to valid range
    idx = np.searchsorted(sorted_d, r_flat, side='right') - 1
    idx = np.clip(idx, 0, len(sorted_d) - 1)
    
    # Best fit item size for each pair
    best_fit_flat = sorted_d[idx]
    best_fit = best_fit_flat.reshape(n, n)
    
    # Closeness: 1 - |r - best_fit| / max(r, 1e-6)
    # Only meaningful where r > 0 and fits
    r_safe = np.where(r_flat.reshape(n, n) > 1e-6, r_flat.reshape(n, n), 1.0)
    closeness = 1.0 - np.abs(residual - best_fit) / r_safe
    closeness = np.maximum(0.0, closeness)
    
    # Only apply where fits and residual > 0 and i != j
    triple_mask = fits & (residual > 0) & ~np.eye(n, dtype=bool)
    triple_bonus = np.where(triple_mask, closeness * 0.2, 0.0)
    
    h += triple_bonus
    
    # Component 4: Size balance bonus - reward pairs where both items are
    # reasonably sized (not one tiny + one huge). This promotes balanced bins.
    # Balance score = 1 - |d[i] - d[j]| / max(d[i], d[j])
    # High when items are similar size, low when very different
    max_d_pair = np.maximum(d[:, None], d[None, :])
    diff = np.abs(d[:, None] - d[None, :])
    balance = 1.0 - diff / np.maximum(max_d_pair, 1e-6)
    balance_bonus = np.where(fits & ~np.eye(n, dtype=bool), balance * 0.15, 0.0)
    
    h += balance_bonus
    
    # Component 5: Residual-target bonus - reward pairs whose residual is
    # close to a specific "target" size that appears frequently in the dataset
    # This encourages packing toward common item sizes
    # Use the median item size as a target
    target_size = np.median(d)
    # Bonus peaks when residual is close to target_size
    residual_dist = np.abs(residual - target_size)
    # Gaussian-like falloff
    sigma = target_size * 0.3 + 1.0
    target_bonus = np.exp(-residual_dist ** 2 / (2 * sigma ** 2)) * 0.1
    target_bonus = np.where(fits & ~np.eye(n, dtype=bool) & (residual > 0), target_bonus, 0.0)
    
    h += target_bonus

    # Final cleanup
    np.fill_diagonal(h, 1e-9)
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9

    return h
# EVOLVE-BLOCK-END
