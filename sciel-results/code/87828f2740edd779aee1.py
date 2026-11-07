import numpy as np
import time

def heuristics(demands, capacity):
    """
    Compute a heuristic matrix for ACO-based bin packing.
    
    The heuristic rewards pairs (i, j) where placing item j after item i
    results in a 'dead gap' — residual space too small for any other item.
    This encourages tight packing. We also incorporate a data-driven
    co-occurrence prior from FFD simulations to identify pairs that
    frequently appear together in good solutions.
    """
    start_time = time.time()
    DEADLINE = 300  # seconds, safety margin before 330s kill
    
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Ensure finite, positive
    demands[~np.isfinite(demands)] = 20.0
    demands[demands <= 0] = 20.0
    demands[demands > capacity] = capacity
    
    # Calculate pair sums and validity
    sums = demands[:, None] + demands[None, :]
    valid = sums <= capacity
    
    # Residual space for each valid pair
    residuals = capacity - sums
    
    # Sort demands for efficient counting
    sorted_demands = np.sort(demands)
    
    # Count how many items fit into each residual (excluding i and j)
    flat_residuals = residuals.flatten()
    total_counts = np.searchsorted(sorted_demands, flat_residuals, side='right')
    total_counts = total_counts.reshape(n, n)
    
    # Exclude i and j from the count
    i_fits = demands[:, None] <= residuals
    j_fits = demands[None, :] <= residuals
    
    dead_gap_counts = total_counts.astype(float) - i_fits.astype(float) - j_fits.astype(float)
    dead_gap_counts = np.maximum(dead_gap_counts, 0)
    
    # Base score: strongly reward dead gaps
    alpha = 2.5
    base_scores = np.exp(-alpha * dead_gap_counts)
    
    # Data-driven co-occurrence prior from FFD simulations
    # Run multiple FFD passes with randomization to estimate co-occurrence
    n_sims = 50
    cooccur = np.zeros((n, n), dtype=float)
    
    # Shuffle indices for randomization
    rng = np.random.RandomState(42)
    
    for sim in range(n_sims):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Randomize the order for FFD
        order = np.argsort(-demands + rng.uniform(0, 5, n))
        
        # First-Fit-Decreasing simulation
        bins = []
        item_to_bin = np.full(n, -1, dtype=int)
        
        for idx in order:
            d = demands[idx]
            placed = False
            for b_idx, bin_items in enumerate(bins):
                if sum(bin_items) + d <= capacity:
                    # Check if any existing item in this bin can form a pair with idx
                    for existing in bin_items:
                        cooccur[existing, idx] += 1
                        cooccur[idx, existing] += 1
                    bin_items.append(d)
                    item_to_bin[idx] = b_idx
                    placed = True
                    break
            if not placed:
                bins.append([d])
                item_to_bin[idx] = len(bins) - 1
    
    # Normalize co-occurrence matrix
    cooccur_max = np.max(cooccur)
    if cooccur_max > 0:
        cooccur = cooccur / cooccur_max
    else:
        cooccur = np.ones((n, n)) / n
    
    # Combine base score with co-occurrence prior
    # Weight: 0.7 for base (structural), 0.3 for co-occurrence (data-driven)
    eta = 0.7 * base_scores + 0.3 * cooccur
    
    # Apply only to valid pairs
    eta[~valid] = 1e-9
    
    # Diagonal should be low (can't place item after itself)
    np.fill_diagonal(eta, 1e-9)
    
    # Ensure all values are positive and finite
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
