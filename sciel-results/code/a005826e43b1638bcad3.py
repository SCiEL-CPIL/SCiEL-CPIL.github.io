import numpy as np
import time

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i
    in the same bin. Larger is more attractive.
    
    Strategy:
    1. Prefer pairs that fit (sum <= capacity).
    2. Among fitting pairs, strongly prefer tight fits (minimal waste).
    3. Use a sharp exponential decay based on waste to distinguish good fits from mediocre ones.
    4. Invalid pairs (sum > capacity) get 1e-9.
    5. Ensure all values are positive and finite.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    
    # Compute pair sums
    pair_sums = d[:, None] + d[None, :]
    
    # Valid mask: pair sum <= capacity
    valid = pair_sums <= capacity
    
    # Initialize heuristic matrix with small positive value for invalid entries
    eta = np.full((n, n), 1e-9, dtype=float)
    
    if not np.any(valid):
        return eta
    
    # For valid entries, compute waste
    # Waste = capacity - pair_sum
    # We want to minimize waste.
    # A common heuristic is 1 / (waste + epsilon) or exp(-waste / scale)
    
    # Let's try a combination:
    # Base: 1.0
    # Penalty for waste: exp(-waste / 10)
    # This gives:
    # waste=0 -> 1.0
    # waste=10 -> 0.37
    # waste=30 -> 0.05
    # waste=50 -> 0.006
    
    # Alternatively, a power law: (1 / (1 + waste)) ** 2
    # waste=0 -> 1.0
    # waste=10 -> 0.008
    # This is very sharp.
    
    # Let's use a tunable decay. The current program uses 1 / (1 + waste/10).
    # waste=0 -> 1.0
    # waste=10 -> 0.5
    # waste=50 -> 0.166
    
    # To improve, we can make the decay sharper to prioritize very tight fits,
    # which is crucial for bin packing to minimize bins.
    # Let's try: 1 / (1 + (waste / 5.0)) ** 2
    # waste=0 -> 1.0
    # waste=5 -> 0.25
    # waste=10 -> 0.111
    # waste=20 -> 0.028
    
    # Or even sharper: exp(-waste / 5)
    # waste=0 -> 1.0
    # waste=5 -> 0.37
    # waste=10 -> 0.135
    # waste=20 -> 0.018
    
    # Let's try a hybrid: strong preference for exact fits, then decay.
    # If waste == 0, give a boost.
    
    waste = capacity - pair_sums
    # Clip waste to non-negative for valid pairs (should be >= 0)
    waste_valid = waste[valid]
    
    # Heuristic value:
    # We want to encourage filling bins completely.
    # A good heuristic is the "best fit" heuristic, which prefers the item that leaves the least space.
    # Since the ant chooses the next item, a high heuristic for (i, j) means "put j after i".
    # This is attractive if i+j is close to capacity.
    
    # Formula: exp(-waste / scale)
    # Let's try scale = 10.
    scale = 10.0
    values = np.exp(-waste_valid / scale)
    
    # To further sharpen, we can square it or use a higher power.
    # values = values ** 2
    
    # Let's try a different approach:
    # The "attractiveness" of placing j after i should be high if the remaining space
    # in the bin after placing j is small, OR if the pair itself is a good fit.
    # The current state is "current item i". The bin has some accumulated weight?
    # No, the heuristic is static: "attractiveness for placing item j right after item i in the same bin".
    # This implies the pair (i, j) is considered. The "current" item is i, and we are placing j.
    # The bin's current load is not explicitly in the heuristic signature, but the ant colony
    # tracks the current item. The heuristic is a precomputed matrix H[i, j].
    # So H[i, j] depends only on i and j.
    # The most logical interpretation is that H[i, j] is high if i and j fit well together.
    
    # Let's use: 1 / (1 + waste/10) ** 2
    # This makes tight fits significantly more attractive.
    values = 1.0 / (1.0 + waste_valid / 10.0) ** 2
    
    eta[valid] = values
    
    # Ensure no non-positive or non-finite values
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
