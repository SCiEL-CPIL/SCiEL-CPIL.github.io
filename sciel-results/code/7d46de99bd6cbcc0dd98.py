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
    diag = np.diag(fits).astype(int)
    deg = np.sum(fits, axis=0) - diag
    
    # 3. Compute scarcity signal
    # Higher scarcity means fewer partners, so the item is "harder" to place
    # We use 1/(deg+1) to avoid division by zero and to normalize
    scarcity = 1.0 / (deg + 1)
    
    # 4. Compute efficiency signal
    # Higher efficiency means the pair fills the bin more completely
    # Normalize by capacity to keep values in a reasonable range (0, 1]
    efficiency = pair_sums / capacity
    
    # 5. Compute complementarity signal
    # For each pair (i, j), measure how well they complement each other
    # A good complement fills the bin without wasting too much space
    # We penalize pairs that leave too much empty space
    gap = capacity - pair_sums
    # Complementarity is high when gap is small relative to capacity
    # But we also want to avoid zero-gap if it makes the bin "tight"
    # Use a smooth function: complementarity = 1 - (gap/capacity)^2
    # This gives higher scores for tighter fits
    complementarity = 1.0 - (gap / capacity) ** 2
    
    # 6. Combine signals
    # The heuristic h[i, j] represents the attractiveness of placing j after i.
    # We combine scarcity of j (global structural signal) with 
    # complementarity (local pair-wise signal)
    # Weight scarcity more heavily to prioritize hard-to-place items
    # H = (2 * scarcity[None, :] + complementarity) / 3
    # But also ensure that efficiency (fill ratio) is considered
    # Final combination: emphasize scarcity and complementarity
    H = (2.0 * scarcity[None, :] * complementarity + complementarity) / 3.0
    
    # 7. Apply mask
    # Only allow transitions if they fit and i != j
    valid = fits & (~np.eye(n, dtype=bool))
    H = np.where(valid, H, 1e-9)
    
    # 8. Ensure all values are positive (minimum 1e-9 for valid entries)
    H = np.maximum(H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
