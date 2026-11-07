"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes a heuristic matrix for the ant colony optimizer.
    
    Strategy: Reward tight packing with emphasis on creating "dead gaps"
    (residual space too small for any remaining item). Uses a smooth surrogate
    for the dead-gap condition combined with tight-fit reward.
    
    The heuristic has three components:
    1. Tight-fit reward: 1/(remaining+1)^2 — sharp preference for minimal waste
    2. Dead-gap bonus: smooth function that peaks when remaining < min_item
    3. Item-size weighting: larger items get slightly more reward for tight fits
    """
    start_time = time.time()
    
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    
    # Calculate pair sums
    pair_sums = d[:, None] + d[None, :]
    
    # Determine which pairs fit in the bin
    fits = pair_sums <= capacity
    
    # Calculate remaining space for fitting pairs
    remaining = capacity - pair_sums
    
    eta = np.full((n, n), 1e-9, dtype=np.float64)
    
    if not np.any(fits):
        return eta
    
    # Minimum item size (for dead-gap detection)
    min_item = float(np.min(d))
    
    # Compute values for fitting pairs only
    # Component 1: Tight-fit reward — sharp preference for minimal waste
    # Use (remaining + 1)^-2 for strong emphasis on tightness
    denom = (remaining + 1.0) ** 2
    tight_reward = np.divide(1.0, denom, out=np.zeros_like(denom), where=denom > 0)
    
    # Component 2: Dead-gap bonus using smooth surrogate
    # When remaining < min_item, no other item can fit → dead gap
    # Smooth surrogate: exp(-remaining / min_item) gives value ~1 when remaining≈0,
    # and decays smoothly. This is a smooth proxy for the nonsmooth indicator.
    # Scale: at remaining=0, value=1; at remaining=min_item, value=exp(-1)≈0.37
    # Use a steeper decay for stronger dead-gap signal: exp(-2*remaining/min_item)
    dead_gap_bonus = np.exp(-2.0 * remaining / min_item)
    
    # Component 3: Item-size weighting — larger items slightly more rewarding
    # Normalize d_j to [0,1] range for weighting
    d_min = float(np.min(d))
    d_max = float(np.max(d))
    d_range = d_max - d_min
    if d_range > 0:
        d_norm = (d - d_min) / d_range  # [0, 1]
    else:
        d_norm = np.zeros_like(d)
    
    # Weight: 1.0 + 0.5 * d_norm → range [1.0, 1.5]
    size_weight = 1.0 + 0.5 * d_norm[None, :]
    
    # Combine: tight_reward * (1 + 2*dead_gap_bonus) * size_weight
    # The dead_gap_bonus amplifies tight fits that also create dead gaps
    values = tight_reward * (1.0 + 2.0 * dead_gap_bonus) * size_weight
    
    # Apply mask: only keep values where fits is True
    eta = np.where(fits, values, 1e-9)
    
    # Sanity check for finite values
    if not np.all(np.isfinite(eta)):
        eta = np.nan_to_num(eta, nan=1e-9, posinf=1e-9, neginf=1e-9)
    
    # Ensure all values are positive (task requirement)
    eta = np.maximum(eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
