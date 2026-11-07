"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    demands = np.array(demands)
    n = len(demands)
    
    # 1. Compute compatibility matrix (fits)
    # fits[i, j] is True if item i and item j can be in the same bin
    pair_sums = demands[:, None] + demands[None, :]
    fits = pair_sums <= capacity
    
    # 2. Compute degree of each item in the compatibility graph
    # deg[j] is the number of items i (i != j) that fit with j
    # Sum along axis 0 gives count for each j
    # Subtract diagonal because an item doesn't fit with itself in this context (i != j)
    diag = np.diag(fits).astype(int)
    deg = np.sum(fits, axis=0) - diag
    
    # 3. Compute scarcity signal (Structural Centrality)
    # Higher scarcity means fewer partners, so the item is "harder" to place
    # This drives the global structural guidance
    scarcity = 1.0 / (deg + 1)
    
    # 4. Compute complementarity signal (Fit Tightness)
    # Instead of raw efficiency, we use the inverse of the remaining gap.
    # This strongly rewards pairs that fill the bin completely,
    # enhancing the "complementarity" between i and j.
    gaps = capacity - pair_sums
    # Add 1.0 to avoid division by zero and to smooth the function
    # For valid fits, gaps >= 0.
    tightness = 1.0 / (gaps + 1.0)
    
    # 5. Combine signals
    # The heuristic h[i, j] represents the attractiveness of placing j after i.
    # We weight the tightness by the scarcity of j.
    # This encourages the ant to place scarce items in positions that
    # maximize bin utilization (minimize gap), prioritizing structural
    # constraints over simple size matching.
    H = tightness * scarcity[None, :]
    
    # 6. Apply mask
    # Only allow transitions if they fit and i != j
    # np.eye(n) has 1s on diagonal
    valid = fits & (~np.eye(n, dtype=bool))
    H = np.where(valid, H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
