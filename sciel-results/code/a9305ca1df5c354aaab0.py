"""Rewritten program for ant colony bin packing heuristic design."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Compute an item-by-item attractiveness matrix for placing item j right after
    item i in the same bin. Larger values indicate more attractive placements.
    
    Strategy: Product-based Attractiveness (Best-Fit proxy).
    
    We define the heuristic entry H[i, j] as the product of the demands of item i 
    and item j, provided they fit in the bin (demands[i] + demands[j] <= capacity).
    If they do not fit, the value is set to a very small epsilon (1e-9).
    
    Why this works:
    1. Feasibility is strictly enforced.
    2. The product d[i] * d[j] acts as a proxy for "packing density". 
       For a fixed item i, the heuristic increases as d[j] increases (up to the 
       feasibility limit). This encourages the ant to pick the largest possible 
       item j that fits, which is a classic "Best Fit" heuristic.
    3. Large items are prioritized because they contribute more to the product. 
       This encourages the colony to pack large items tightly together, which is 
       crucial for minimizing the number of bins in bin packing problems.
    """
    demands = np.asarray(demands, dtype=np.float64)
    n = len(demands)
    
    # Create pairwise sums to check feasibility
    pair_sums = demands[:, None] + demands[None, :]
    
    # Feasibility mask: True if the pair fits in the bin
    feasible = pair_sums <= capacity
    
    # Calculate the product of demands for all pairs
    # This rewards pairs where both items are large, and for a fixed i, 
    # rewards larger j (tighter fit).
    products = demands[:, None] * demands[None, :]
    
    # Initialize heuristic with a very small value (epsilon) for infeasible pairs
    # The problem statement says "Heuristic entries that are not positive become 1e-9".
    # We ensure feasible pairs have positive values (which they do, since demands >= 20).
    heuristic = np.where(feasible, products, 1e-9)
    
    # Ensure all values are positive and finite
    # Products of positive demands are positive. 
    # 1e-9 is positive.
    # We clamp just in case of any numerical issues, though not expected here.
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
