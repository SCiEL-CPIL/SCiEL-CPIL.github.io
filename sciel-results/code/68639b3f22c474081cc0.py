"""
Parameterized heuristic optimization for Ant Colony Bin Packing.
Combines a product-based item importance model with an inverse-waste tightness model,
tuned via a fast greedy simulation proxy.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix using a parameterized model tuned by a fast greedy simulation.
    
    Model: H[i,j] = (d_i * d_j) / (waste_ij + eps)^gamma
    - d_i * d_j: Favors pairing large items, which are critical for bin packing efficiency.
    - (waste_ij + eps)^-gamma: Favors tight fits. gamma controls the strength of this preference.
    
    We grid-search gamma to minimize the number of bins in a fast greedy simulation that 
    mimics the ACO construction process.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    
    if n == 0:
        return np.empty((0, 0))
    
    # Precompute pair properties
    sums = d[:, None] + d[None, :]
    fits = sums <= capacity
    waste = capacity - sums
    
    # For non-fitting pairs, waste is negative or large. We only care about fitting pairs.
    # For fitting pairs, waste >= 0.
    # We use a small epsilon to avoid division by zero and to cap the maximum value.
    epsilon = 1.0
    
    # Grid search over gamma
    # gamma = 0: H ~ product (size importance only)
    # gamma = 1: H ~ product / waste (balanced)
    # gamma > 1: Strong preference for tight fits
    gamma_candidates = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    
    best_H = None
    best_bins = np.inf
    
    # Pre-sort indices for the simulation to speed up "largest remaining" selection
    # In the simulation, we will maintain a list of remaining items.
    
    def simulate_greedy_bins(H):
        """
        Fast greedy simulation to estimate ACO performance.
        Strategy:
        1. Maintain a set of remaining items.
        2. Start a bin with the largest remaining item.
        3. Repeatedly select the next item j that maximizes H[current, j] among remaining items that fit with current.
        4. If no such item exists, close bin and start new one with largest remaining.
        5. Return total bins used.
        """
        remaining = list(range(n))
        bins_used = 0
        
        while remaining:
            # Start new bin with largest item
            # Find index of largest item in remaining
            max_idx_in_rem = 0
            max_val = -1
            for k in range(len(remaining)):
                if d[remaining[k]] > max_val:
                    max_val = d[remaining[k]]
                    max_idx_in_rem = k
            
            current_item = remaining.pop(max_idx_in_rem)
            bins_used += 1
            
            # Try to add more items to this bin
            while True:
                best_next = -1
                best_score = -1e18
                # Iterate over remaining items to find best fit
                # Note: This is O(n^2) per bin, total O(n^3) worst case, but n=120 is small.
                # Optimization: We can break early if we find a perfect fit? No, heuristic might prefer other.
                for idx, j in enumerate(remaining):
                    if H[current_item, j] > best_score:
                        # Check if it fits (H should be 1e-9 if not, but double check for safety)
                        if d[current_item] + d[j] <= capacity:
                            best_score = H[current_item, j]
                            best_next = idx
                        else:
                            # If H is high but doesn't fit, it's a bug in H construction, 
                            # but our H construction masks non-fits to 1e-9.
                            pass
                
                if best_next == -1:
                    break
                
                # Place the best next item
                current_item = remaining.pop(best_next)
                
        return bins_used

    # Run grid search
    for gamma in gamma_candidates:
        if time.time() > deadline - 10:
            break
            
        # Compute H for this gamma
        # H[i,j] = (d_i * d_j) / (waste_ij + eps)^gamma
        # For non-fitting, H = 1e-9
        
        # Vectorized computation
        products = d[:, None] * d[None, :]
        
        # safe_waste: for fitting pairs, waste >= 0. For non-fitting, we don't care.
        # We calculate term only for fitting pairs to avoid issues with negative waste.
        raw_H = np.full((n, n), 1e-9, dtype=np.float64)
        
        if gamma == 0.0:
            tightness = np.ones((n, n), dtype=np.float64)
        else:
            # Only compute for fitting pairs to avoid negative base for fractional powers
            # waste is capacity - sums. For fits, waste >= 0.
            # We add epsilon to avoid div by zero.
            # (waste + eps)^-gamma
            # Use np.power for vectorization
            # Mask non-fitting to avoid invalid ops or just compute and mask later
            # Since waste can be 0, (0+eps)^-gamma is large but finite.
            
            # To be safe and fast, compute only on mask
            # But creating a masked array might be slower. Let's compute full and mask.
            # For non-fitting, waste < 0. (negative)^gamma is undefined for fractional gamma.
            # So we MUST mask before power.
            
            safe_waste = np.where(fits, waste + epsilon, 1.0) # 1.0 for non-fits, doesn't matter
            tightness = np.power(safe_waste, -gamma)
            
        raw_H[fits] = products[fits] * tightness[fits]
        
        # Ensure finiteness
        raw_H[~np.isfinite(raw_H)] = 1e-9
        
        # Evaluate with simulation
        bins = simulate_greedy_bins(raw_H)
        
        if bins < best_bins:
            best_bins = bins
            best_H = raw_H.copy()
            
        # Early termination if we get a very good result? 
        # Theoretical lower bound is ceil(sum(d)/capacity).
        lb = int(np.ceil(np.sum(d) / capacity))
        if best_bins == lb:
            break

    # Fallback if something went wrong
    if best_H is None:
        # Default to gamma=1.0
        products = d[:, None] * d[None, :]
        safe_waste = np.where(fits, waste + epsilon, 1.0)
        tightness = np.power(safe_waste, -1.0)
        best_H = np.full((n, n), 1e-9, dtype=np.float64)
        best_H[fits] = products[fits] * tightness[fits]
        best_H[~np.isfinite(best_H)] = 1e-9

    # Final shape check
    if best_H.shape != (n, n):
        raise ValueError("Wrong shape")
        
    return best_H
# EVOLVE-BLOCK-END
