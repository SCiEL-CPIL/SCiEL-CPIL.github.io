"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    demands = np.array(demands)
    
    # Calculate pair sums
    pair_sums = demands[:, None] + demands[None, :]
    
    # Identify valid pairs
    valid = pair_sums <= capacity
    
    # Calculate gaps
    gaps = capacity - pair_sums
    
    # Find the minimum item size
    min_d = np.min(demands)
    
    # Initialize heuristic matrix with a very small positive value for invalid pairs
    # The problem states non-positive becomes 1e-9, so we can use 1e-9 directly
    eta = np.full((n, n), 1e-9)
    
    # Identify different types of valid pairs
    # 1. Perfect fits (gap == 0)
    perfect = valid & (gaps == 0)
    
    # 2. Useful gaps (gap >= min_d)
    # Note: gap == 0 is already handled, so strictly > 0 is implied if we exclude perfect?
    # Actually, if min_d > 0, gap >= min_d implies gap > 0.
    useful = valid & (gaps >= min_d)
    
    # 3. Dead gaps (0 < gap < min_d)
    dead = valid & (gaps < min_d) & (gaps > 0)
    
    # Assign scores
    # Perfect fits get a high score
    eta[perfect] = 1000.0
    
    # Useful gaps get a score inversely proportional to the gap size
    # We add 1 to avoid division by zero, though gap >= min_d >= 20 (since demands are 20-99)
    # So gaps are at least 20.
    # Score range: 100 / (20+1) ~ 4.7 to 100 / (99+1) ~ 1.0 (max gap is 150-40=110? No, min sum is 40, max gap 110)
    # Wait, demands are 20-99. Min sum 40. Max gap 110.
    # So useful gaps are 20 to 110.
    eta[useful] = 100.0 / (gaps[useful] + 1.0)
    
    # Dead gaps get a low constant score, worse than any useful gap
    # The smallest useful gap is min_d (20). Score ~ 100/21 ~ 4.7.
    # So dead gaps should be < 4.7. Let's use 1.0.
    eta[dead] = 1.0
    
    return eta
# EVOLVE-BLOCK-END
