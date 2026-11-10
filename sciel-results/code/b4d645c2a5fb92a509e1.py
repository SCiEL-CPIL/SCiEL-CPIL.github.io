"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Compute an item-by-item attractiveness matrix for placing item j right after
    item i in the same bin. Larger values mean more attractive.
    
    Combines:
    1. A smooth Gaussian peak centered at perfect fill (ratio = 1.0)
    2. An inverse-waste term that strongly favors tight fits
    3. A geometric-mean bonus that breaks ties by preferring larger items
    4. A size-weighted complementarity signal for hard-to-fit items
    """
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    cap = float(capacity)
    
    # Pairwise sums
    pair = d[:, None] + d[None, :]
    
    # Check which pairs fit in the bin
    fits = pair <= cap
    
    # Remaining space (waste) for fitting pairs
    remaining = cap - pair
    
    # Safe remaining for computation (avoid issues with non-fitting pairs)
    safe_remaining = np.where(fits, remaining, cap)  # non-fitting gets large waste
    
    # --- Component 1: Gaussian peak centered at perfect fill ---
    ratio = np.where(fits, pair / cap, 0.0)
    sigma = 8.0
    gaussian = np.where(fits, np.exp(-sigma * (1.0 - ratio) ** 2), 0.0)
    
    # --- Component 2: Inverse waste term (sharp decay for loose fits) ---
    # 1 / (1 + (remaining / tau)^2) with tau=15 for moderate sharpness
    tau = 15.0
    inv_waste = np.where(fits, 1.0 / (1.0 + (safe_remaining / tau) ** 2), 0.0)
    
    # --- Component 3: Geometric mean bonus (tiebreaker for larger items) ---
    geo_mean = np.sqrt(d[:, None] * d[None, :])
    geo_norm = geo_mean / cap  # normalized to [~0.13, ~0.66]
    geo_bonus = np.where(fits, geo_norm, 0.0)
    
    # --- Component 4: Size-weighted complementarity ---
    # Reward pairs where the smaller item is large (hard-to-fit items paired well)
    min_item = np.minimum(d[:, None], d[None, :])
    min_norm = min_item / cap  # in [~0.13, ~0.66]
    # Complementarity: how well the two items complement each other
    # High when both are large and their sum is close to capacity
    complementarity = np.where(fits, min_norm * inv_waste, 0.0)
    
    # Combine all components with tuned weights
    w_gauss = 0.35
    w_inv = 0.35
    w_geo = 0.15
    w_comp = 0.15
    
    combined = (w_gauss * gaussian + 
                w_inv * inv_waste + 
                w_geo * geo_bonus + 
                w_comp * complementarity)
    
    # Ensure positive values for fitting pairs, small value for non-fitting
    result = np.where(fits, combined, 1e-9)
    
    # Guarantee minimum positive value and finite values
    result = np.maximum(result, 1e-9)
    result[~np.isfinite(result)] = 1e-9
    
    return result
# EVOLVE-BLOCK-END
