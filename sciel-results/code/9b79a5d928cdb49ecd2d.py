"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 120.0  # seconds
_start_time = time.time()


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Safety: fall back to simple best fit
        return -(bins - item)
    
    remaining = bins - item
    feasible = remaining >= 0
    
    # Smooth surrogate for the nonsmooth min/max objective:
    # Use a soft-min via log-sum-exp to create a smooth approximation
    # of the "tightest fit" that avoids discontinuities at the feasible boundary.
    safe_bins = np.where(feasible & (bins > 0), bins, 1.0)
    
    # Fill ratio: how much of available space is used
    fill_ratio = item / safe_bins
    
    # Best-fit component: prefer smaller remaining after placement
    best_fit = -remaining
    
    # Quadratic term: strongly penalize large gaps (reduces fragmentation)
    quad_penalty = -(remaining ** 2) / 10000.0
    
    # Cubic term: even stronger penalty for very large gaps
    cubic_penalty = -(remaining ** 3) / 1000000.0
    
    # Smooth surrogate: use log-sum-exp to create a smooth "soft-min" of remaining
    # This creates a smooth approximation that encourages tight fits without
    # the discontinuity of a hard min
    # Scale remaining to be in a reasonable range for log-sum-exp
    scaled_remaining = remaining / 10.0
    # Add a small constant to avoid log(0)
    scaled_remaining_safe = np.where(scaled_remaining > 0.001, scaled_remaining, 0.001)
    # Log-sum-exp approximation of the minimum (soft-min)
    # For a set of values, log(sum(exp(x_i))) approximates max(x_i)
    # So -log(sum(exp(-x_i))) approximates min(x_i)
    # We want to prefer small remaining, so we use a soft-min of remaining
    # Higher score = better, so we want the soft-min of remaining to be small
    # Therefore, we use -soft_min(remaining) as a component
    # But we want this to be smooth, so we use a temperature parameter
    temp = 1.0
    soft_min_remaining = -temp * np.log(np.sum(np.exp(-scaled_remaining_safe / temp)))
    
    # Polarization component: encourage bins to become either full or empty
    # This reduces the number of 'half-full' bins that are difficult to pack efficiently
    # A bin is "polarized" if it's close to full (remaining close to 0) or close to empty
    # We measure polarization by how far remaining is from the "middle" (capacity/2)
    # Bins that are very full or very empty get higher polarization scores
    capacity = 100.0
    middle = capacity / 2.0
    # Distance from middle: larger distance = more polarized
    # We want to prefer polarized bins, so we add this as a positive component
    # But we only want to reward polarization for feasible bins
    polarization = np.abs(remaining - middle) / middle
    # Scale to be comparable to other terms
    polarization_scaled = polarization * 0.5
    
    # Composite score: weighted sum of all components
    # Higher fill_ratio, smaller remaining (best_fit), smaller quadratic/cubic penalties,
    # smaller soft-min remaining, and higher polarization all contribute positively
    scores = np.where(
        feasible,
        fill_ratio + 
        best_fit / 100.0 + 
        quad_penalty + 
        cubic_penalty + 
        soft_min_remaining / 10.0 + 
        polarization_scaled,
        -np.inf
    )
    
    return scores
# EVOLVE-BLOCK-END
