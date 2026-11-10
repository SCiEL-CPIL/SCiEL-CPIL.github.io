"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    d = demands.astype(float)
    pair = d[:, None] + d[None, :]
    
    # Only consider pairs that fit in the bin
    fits = pair <= capacity
    
    # For fitting pairs, compute how well they fill the bin
    # We want to strongly favor pairs that nearly fill the bin
    # Use a sharp peak at capacity: (pair/capacity)^power where power is large
    # This creates strong preference for near-perfect fits
    
    ratio = np.where(fits, pair / capacity, 0.0)
    
    # Sharp decay: use high power to strongly favor near-perfect fits
    # Power of 6 creates very sharp preference for ratios close to 1.0
    sharp = np.where(fits, ratio ** 6, 0.0)
    
    # Ensure positive values for fitting pairs, small value for non-fitting
    result = np.where(fits, sharp, 1e-9)
    
    # Guarantee minimum positive value
    result = np.maximum(result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
