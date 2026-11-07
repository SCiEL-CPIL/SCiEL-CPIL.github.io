"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    demands = np.array(demands, dtype=np.float64)
    
    # Calculate pair sums
    pair_sums = demands[:, None] + demands[None, :]
    
    # Identify valid pairs (pair sum <= capacity)
    valid = pair_sums <= capacity
    
    # Calculate gaps (remaining space in bin after placing both items)
    gaps = capacity - pair_sums
    
    # Find the minimum item size
    min_d = np.min(demands)
    
    # Initialize heuristic matrix with a very small positive value for invalid pairs
    eta = np.full((n, n), 1e-9)
    
    # Identify categories of valid pairs
    perfect = valid & (gaps == 0)
    useful = valid & (gaps >= min_d) & (gaps > 0)
    dead = valid & (gaps < min_d) & (gaps > 0)
    
    # Perfect fits: highest priority
    eta[perfect] = 10000.0
    
    # Useful gaps: reward tighter fits more heavily
    # Use exponential decay to strongly prefer smaller gaps
    # gap ranges from min_d (~20) to max (~110)
    # Scale so that gap=min_d gets high score, larger gaps get progressively lower
    if np.any(useful):
        # Use a power law: score = base / (gap + offset)^power
        # With power > 1, we create stronger discrimination between small and large gaps
        useful_gaps = gaps[useful]
        eta[useful] = 500.0 / (useful_gaps + 5.0) ** 1.5
    
    # Dead gaps: these waste capacity, so give them a low but non-trivial score
    # They are worse than any useful gap but better than invalid (1e-9)
    # The smallest useful gap score is ~ 500/(20+5)^1.5 = 500/125 = 4.0
    # So dead gaps should be significantly below that
    if np.any(dead):
        dead_gaps = gaps[dead]
        # Linear interpolation: smaller dead gaps get slightly higher scores
        # Max dead gap is min_d - 1, min is 1
        # Score range: 0.1 to 0.5
        norm_dead = dead_gaps / (min_d - 1.0) if min_d > 1 else 0.5
        eta[dead] = 0.5 - 0.4 * norm_dead
    
    # Ensure all values are positive
    eta = np.maximum(eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
