import numpy as np
import time

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i.
    Larger values are more attractive.
    
    Strategy: Combine the smooth bounded structure of the current program with
    sharper decay insights. We use a normalized waste metric and a sharp power
    to prioritize tight fits. Additionally, we incorporate a size-weighted
    complementarity term to encourage pairing large items (which are harder to
    place) with complementary items that fill the bin tightly.
    """
    start_time = time.time()
    
    d = np.asarray(demands, dtype=float)
    n = len(d)
    
    if n == 0:
        return np.empty((0, 0), dtype=float)

    cap = float(capacity)
    
    # Pairwise sums
    pair_sum = d[:, None] + d[None, :]
    
    # Check which pairs fit
    fits = pair_sum <= cap
    
    # Calculate waste (remaining space)
    waste = cap - pair_sum
    
    # Normalize waste by capacity to make the scale parameter dimensionless
    # waste_norm is in [0, 1] for fitting pairs
    waste_norm = waste / cap
    
    # Initialize with a very low value for non-fitting pairs
    h = np.full((n, n), 1e-9, dtype=float)
    
    # For fitting pairs, apply the heuristic
    # Base term: sharp decay based on normalized waste
    # h_base = 1 / (1 + (waste_norm / scale)^power)
    scale = 0.1
    power = 3.0
    
    term = (waste_norm / scale) ** power
    h_base = 1.0 / (1.0 + term)
    
    # Size-weighted complementarity term
    # We want to boost pairs where the sum is large (close to capacity) 
    # and the items are individually large.
    # Let's use the fraction of capacity filled: fill_ratio = pair_sum / cap
    # And a size factor: maybe (d_i + d_j) / (2 * cap) which is same as fill_ratio/2?
    # Or simply use pair_sum directly as a weight.
    
    # Let's try: h = h_base * (pair_sum / cap)
    # This rewards pairs that fill the bin more.
    # For waste=0 (pair_sum=cap), fill_ratio=1, h = 1 * 1 = 1.
    # For waste=15 (pair_sum=135), fill_ratio=0.9, h_base=0.5, h = 0.45.
    # For waste=75 (pair_sum=75), fill_ratio=0.5, h_base=0.008, h = 0.004.
    
    # This seems reasonable. It emphasizes tight fits AND large sums.
    
    h_weighted = h_base * (pair_sum / cap)
    
    # Apply mask: if it doesn't fit, set to 1e-9
    h = np.where(fits, h_weighted, 1e-9)
    
    # Safety checks: ensure no non-finite values or non-positive values
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 290:
        # Return the best result found so far (which is h)
        pass
        
    return h
