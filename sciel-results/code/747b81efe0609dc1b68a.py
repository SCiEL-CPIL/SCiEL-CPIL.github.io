import numpy as np

def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Pair sums
    pair_sum = demands[:, None] + demands[None, :]
    
    # Compatibility mask
    fits = pair_sum <= capacity
    
    # Degrees: number of compatible partners for each item
    # fits[i, j] is True if i and j are compatible.
    # deg[j] is the number of i's such that fits[i, j] is True.
    deg = fits.sum(axis=0)
    
    # Scarcity score: higher for items with fewer partners
    # Using deg + 1 to avoid division by zero and to scale
    scarcity = 1.0 / (deg + 1)
    
    # Tightness score: higher for pairs that fill the bin more
    tightness = pair_sum / capacity
    
    # Combine scores
    # We want to favor pairs that are tight AND involve a scarce item j
    # H[i, j] = tightness[i, j] * scarcity[j]
    heur = tightness * scarcity[None, :]
    
    # Apply mask
    heur = np.where(fits, heur, 1e-9)
    
    return heur
