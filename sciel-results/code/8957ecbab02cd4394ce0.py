"""Rewritten program with a stronger heuristic for ant colony optimization."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ant colony optimization.
    
    The heuristic encourages placing items that fit together tightly in the same bin.
    For a pair (i, j) that fits (demands[i] + demands[j] <= capacity), the attractiveness
    is proportional to the sum of their sizes divided by the residual space plus one.
    This creates a strong incentive to minimize empty space in the bin.
    """
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Calculate the sum of sizes for all pairs
    pair_sums = demands[:, None] + demands[None, :]
    
    # Initialize the heuristic matrix with a very small value for non-fitting pairs
    H = np.full((n, n), 1e-9, dtype=float)
    
    # Identify pairs that fit in the bin
    # Note: capacity is a scalar
    fits = pair_sums <= capacity
    
    # Calculate heuristic values for fitting pairs
    # Formula: (sum) / (capacity - sum + 1)
    # This is always positive for fitting pairs.
    # The +1 prevents division by zero for perfect fits and ensures stability.
    
    # We only compute for the valid entries to avoid unnecessary calculations
    # or warnings, though numpy handles broadcasting well.
    if np.any(fits):
        # Residual space + 1
        denom = capacity - pair_sums[fits] + 1.0
        # Sum of sizes
        numer = pair_sums[fits]
        
        H[fits] = numer / denom
        
    return H
# EVOLVE-BLOCK-END
