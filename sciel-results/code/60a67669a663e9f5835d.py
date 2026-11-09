"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin using a refined Best Fit heuristic with explicit sliver avoidance.
    
    The core strategy is Best Fit (preferring bins with least remaining capacity), 
    implemented via a power-law score: -(remaining/capacity)^alpha.
    
    To address fragmentation caused by 'slivers' (very small remaining spaces that 
    are unusable for most future items), we add a smooth penalty for remainders 
    below a threshold. This effectively creates a 'dead zone' near zero, pushing 
    the algorithm to prefer slightly looser fits if the tight fit would result in 
    an unusable sliver. This balances the tight-packing efficiency of Best Fit 
    with the robustness of avoiding dead space, which is particularly effective 
    for Weibull-distributed items where many small items can fill gaps but 
    extremely small gaps are often wasted.
    
    alpha = 1.2: Slightly more aggressive than linear Best Fit, ensuring tight 
    preference for moderate remainders.
    Penalty: Smooth sigmoid-like penalty for rem < 5% of capacity.
    """
    start_time = time.time()
    
    # Compute remaining capacity if item is placed in each bin
    remaining = bins - item
    
    # Only consider bins with enough remaining capacity
    valid_mask = remaining >= 0
    
    # Create scores array, default to very low for invalid bins
    scores = np.full(len(bins), -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    rem_valid = remaining[valid_mask]
    
    # Capacity is 100 per the task description
    capacity = 100.0
    
    # Normalize remaining by capacity to make the transform scale-invariant
    rem_norm = rem_valid / capacity
    
    # Primary: Nonlinear Best Fit preference
    # alpha > 1 makes the function flatter for large remainders (less penalty for 
    # loose fits) but still prefers tight fits.
    # We use alpha=1.2 to be slightly more sensitive to tightness than linear, 
    # but the penalty below will counteract this for very small remainders.
    alpha = 1.2
    primary_score = -(rem_norm ** alpha)
    
    # Secondary: Smooth penalty for 'slivers' (unusable small remainders)
    # We penalize remainders below 5% of capacity.
    # Using a smooth sigmoid to avoid discontinuities.
    # penalty = A / (1 + exp((rem_norm - threshold) / scale))
    # When rem_norm << threshold, penalty ~ A.
    # When rem_norm >> threshold, penalty ~ 0.
    threshold = 0.05
    scale = 0.02
    A = 0.15  # Strength of the sliver penalty
    
    # Calculate penalty
    # Clip arguments to avoid overflow in exp
    x = (rem_norm - threshold) / scale
    x = np.clip(x, -50, 50)
    penalty = A / (1.0 + np.exp(x))
    
    # Combined score: Primary (tight fit) minus Penalty (sliver avoidance)
    # Note: Penalty is positive, so we subtract it to lower the score for slivers.
    scores[valid_mask] = primary_score - penalty
    
    # Deadline guard: if we're close to timeout, return immediately
    elapsed = time.time() - start_time
    if elapsed > 140:  # Safety margin
        return scores
    
    return scores
# EVOLVE-BLOCK-END
