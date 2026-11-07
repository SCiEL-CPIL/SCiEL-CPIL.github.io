"""Rewritten program for ant colony bin packing heuristic."""
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
    
    # For fitting pairs, compute attractiveness based on:
    # 1. Fill ratio (higher is better)
    # 2. Inverse remaining space (tighter packing is better)
    # 3. Residual quality (leftover that can fit another item)
    
    # Component 1: Fill ratio squared - rewards fuller packs quadratically
    fill_ratio = pair / capacity
    fill_score = np.where(fits, fill_ratio ** 2, 0.0)
    
    # Component 2: Inverse remaining space - rewards tight packing
    # remaining = capacity - pair (when fits)
    remaining = np.where(fits, capacity - pair, capacity)
    # Avoid division by zero; use a small epsilon
    inv_remaining = np.where(remaining > 1e-6, 1.0 / (remaining + 1.0), 0.0)
    # Scale to be comparable to fill_score
    inv_remaining_score = np.where(fits, inv_remaining * capacity * 0.5, 0.0)
    
    # Component 3: Residual quality - check if leftover can fit another item
    # For each fitting pair (i,j), residual = capacity - d[i] - d[j]
    # If residual >= min_item_size, it can fit at least one more item
    min_d = d.min()
    max_d = d.max()
    
    # Residual is "useful" if it's large enough to fit another item
    # or very small (tight pack)
    residual = np.where(fits, capacity - pair, 0.0)
    
    # Useful residual: either can fit another item (residual >= min_d)
    # or is very tight (residual < min_d * 0.3)
    useful_residual = np.where(fits, 
                                (residual >= min_d) | (residual < min_d * 0.3), 
                                False)
    residual_bonus = np.where(useful_residual, 0.3, 0.0)
    
    # Combine components
    # h[i,j] = fill_score + inv_remaining_score + residual_bonus
    h = np.where(fits, fill_score + inv_remaining_score + residual_bonus, 1e-9)
    
    # Don't allow placing an item after itself
    np.fill_diagonal(h, 1e-9)
    
    # Ensure non-positive and non-finite values become 1e-9
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9
    
    # Check deadline
    if time.time() > deadline:
        return h
    
    # Optional: Add a small bonus for pairs where the residual closely matches
    # some other item's size, enabling a triple fit
    # This is O(n^2 * log(n)) with binary search
    sorted_d = np.sort(d)
    
    # For efficiency, only do this if we have time
    if time.time() < deadline - 30:
        # For each fitting pair, find the item size closest to the residual
        # and add a bonus if it's close
        bonus = np.zeros((n, n), dtype=float)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if not fits[i, j]:
                    continue
                r = residual[i, j]
                if r <= 0:
                    continue
                # Find the largest item that fits into residual r
                # Use binary search on sorted_d
                idx = np.searchsorted(sorted_d, r, side='right') - 1
                if idx >= 0:
                    best_fit = sorted_d[idx]
                    # Bonus based on how well it fits
                    # If best_fit is close to r, that's good
                    closeness = 1.0 - abs(r - best_fit) / max(r, 1e-6)
                    closeness = max(0.0, closeness)
                    bonus[i, j] = closeness * 0.2
        
        h += bonus
        np.fill_diagonal(h, 1e-9)
        h[~np.isfinite(h)] = 1e-9
        h[h <= 0] = 1e-9
    
    return h
# EVOLVE-BLOCK-END
