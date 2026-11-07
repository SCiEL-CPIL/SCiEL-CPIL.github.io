"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def heuristics(demands, capacity):
    """
    Compute an item-pair attractiveness matrix for ant colony optimization
    of offline bin packing.
    
    This version uses a hybrid heuristic:
    1. Static geometric features (fill ratio, residual quality).
    2. Empirical co-occurrence prior from lightweight randomized Best-Fit simulations.
    """
    t0 = time.time()
    deadline = t0 + 290.0  # Safety margin

    n = len(demands)
    d = np.array(demands, dtype=float)
    
    # 1. Static Component Calculation
    pair = d[:, None] + d[None, :]
    fits = pair <= capacity
    
    # Initialize static heuristic
    h_static = np.zeros((n, n), dtype=float)
    
    # Fill ratio squared
    fill_ratio = np.where(fits, pair / capacity, 0.0)
    fill_score = fill_ratio ** 2
    
    # Inverse remaining space
    remaining = np.where(fits, capacity - pair, capacity)
    inv_remaining = np.where(remaining > 1e-6, 1.0 / (remaining + 1.0), 0.0)
    inv_remaining_score = np.where(fits, inv_remaining * capacity * 0.5, 0.0)
    
    # Residual quality
    min_d = d.min()
    residual = np.where(fits, capacity - pair, 0.0)
    useful_residual = np.where(fits, (residual >= min_d) | (residual < min_d * 0.3), False)
    residual_bonus = np.where(useful_residual, 0.3, 0.0)
    
    h_static = np.where(fits, fill_score + inv_remaining_score + residual_bonus, 0.0)
    
    # Normalize static heuristic to [0, 1] for blending
    max_static = h_static.max()
    if max_static > 0:
        h_static = h_static / max_static
    
    # 2. Empirical Component Calculation (Co-occurrence Prior)
    # Run lightweight randomized Best-Fit simulations
    co_occurrence = np.zeros((n, n), dtype=float)
    
    # Use a fixed seed for reproducibility in the heuristic generation
    # The ACO itself is seeded per instance, but the heuristic is static per instance.
    # We want the heuristic to be robust, so we average over multiple random runs.
    num_sims = 50
    
    for _ in range(num_sims):
        # Random permutation of items
        perm = list(range(n))
        random.shuffle(perm)
        
        bins = [] # List of lists of item indices
        bin_remaining = [] # List of remaining capacities
        
        for idx in perm:
            item_size = d[idx]
            best_bin_idx = -1
            best_remaining = float('inf')
            
            # Find Best Fit bin
            for b_idx in range(len(bins)):
                if bin_remaining[b_idx] >= item_size:
                    if bin_remaining[b_idx] < best_remaining:
                        best_remaining = bin_remaining[b_idx]
                        best_bin_idx = b_idx
            
            if best_bin_idx == -1:
                # Open new bin
                bins.append([idx])
                bin_remaining.append(capacity - item_size)
            else:
                # Place in best fit bin
                bins[best_bin_idx].append(idx)
                bin_remaining[best_bin_idx] -= item_size
        
        # Update co-occurrence counts for consecutive items in each bin
        for b in bins:
            for k in range(len(b) - 1):
                i = b[k]
                j = b[k+1]
                co_occurrence[i, j] += 1
    
    # Normalize co-occurrence to [0, 1]
    max_co = co_occurrence.max()
    if max_co > 0:
        co_occurrence = co_occurrence / max_co
    
    # 3. Combine Static and Empirical Components
    # Weighting: Empirical prior is usually stronger if simulations are good.
    # Let's try 0.5 * static + 0.5 * empirical, or adjust based on "confidence".
    # Since n=120, 50 sims give ~50*119 pairs observed. Many pairs might be 0.
    # We should only add empirical bonus if the pair was observed, or handle zeros carefully.
    
    # Blend
    alpha = 0.6 # Weight for static
    beta = 0.4  # Weight for empirical
    
    h = alpha * h_static + beta * co_occurrence
    
    # Ensure fitting pairs have a minimum attractiveness
    # Non-fitting pairs remain 0 (will be masked later or set to 1e-9)
    # The problem states: "Heuristic entries that are not positive become 1e-9"
    
    # Apply mask for fitting pairs
    h = np.where(fits, h, 0.0)
    
    # Ensure diagonal is low (don't place item after itself)
    np.fill_diagonal(h, 0.0)
    
    # Convert non-positive/non-finite to 1e-9
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9
    
    # Check deadline
    if time.time() > deadline:
        return h

    # Optional: Add small bonus for complementarity (triple fit potential)
    # This is similar to the previous version but kept light
    if time.time() < deadline - 5:
        sorted_d = np.sort(d)
        residual = np.where(fits, capacity - pair, 0.0)
        
        # Vectorized searchsorted for best fit item in residual
        idx = np.searchsorted(sorted_d, residual, side='right') - 1
        idx = np.clip(idx, 0, len(sorted_d) - 1)
        best_fit = sorted_d[idx]
        
        valid_mask = fits & (residual > 0)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            closeness = 1.0 - np.abs(residual - best_fit) / np.maximum(residual, 1e-6)
        
        closeness = np.where(valid_mask, np.maximum(closeness, 0.0), 0.0)
        
        # Add small bonus
        bonus = closeness * 0.1
        h += bonus
        
        # Re-apply constraints
        h = np.where(fits, h, 1e-9)
        np.fill_diagonal(h, 1e-9)
        h[~np.isfinite(h)] = 1e-9
        h[h <= 0] = 1e-9

    return h
# EVOLVE-BLOCK-END
