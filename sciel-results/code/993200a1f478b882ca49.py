"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    # Initialize co-occurrence matrix
    C = np.zeros((n, n), dtype=np.float64)
    
    # Run multiple randomized First-Fit simulations to build a prior
    # This captures global structural affinities (community structures)
    num_runs = 100
    for seed in range(num_runs):
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        
        bins = []
        current_bin = []
        current_sum = 0.0
        
        for idx in perm:
            d = demands[idx]
            if current_sum + d <= capacity:
                current_sum += d
                current_bin.append(idx)
            else:
                if current_bin:
                    bins.append(current_bin)
                current_bin = [idx]
                current_sum = d
        
        if current_bin:
            bins.append(current_bin)
            
        # Update co-occurrence counts
        for b in bins:
            for i_idx in b:
                for j_idx in b:
                    if i_idx != j_idx:
                        C[i_idx, j_idx] += 1.0

    # Compute pair sums and gaps
    pair_sum = demands[:, None] + demands[None, :]
    gap = capacity - pair_sum
    
    # Initialize heuristic matrix
    H = np.full((n, n), 1e-9)
    
    # Valid pairs are those where gap >= 0
    valid = gap >= 0
    
    # For valid pairs, combine gap penalty (rewarding small gaps/tight packing)
    # with co-occurrence prior
    # H = (1 + C) / (gap + 1)
    # We add 1 to gap to avoid division by zero and to scale
    # We add 1 to C to ensure positive weights even for unobserved pairs
    
    # Only compute for valid pairs to avoid unnecessary operations
    # We need to be careful with the diagonal (i == i) as it's not used in placement
    # but the problem implies a matrix for placing j after i.
    
    # Vectorized computation
    # Create a mask for valid and i != j? 
    # The ACO loop picks j != current. So diagonal doesn't strictly matter for probability, 
    # but let's keep it safe.
    
    # Calculate the heuristic value for valid pairs
    # We want to avoid division by zero if gap is -1 (impossible since valid means gap>=0, min gap 0)
    # gap + 1 is at least 1.
    
    # We'll compute the full matrix formula and then mask
    # To save time, we can just compute on the valid mask
    # But vectorized operations on the whole matrix are fast for n=120
    
    # Let's use np.where
    # numerator = 1.0 + C
    # denominator = gap + 1.0
    # H_calc = numerator / denominator
    
    # However, if gap < 0, we want 1e-9.
    # If gap >= 0, we want H_calc.
    
    # Note: C is symmetric. H should probably be symmetric or at least consistent.
    # The ACO uses H[current, j].
    
    H_valid = (1.0 + C) / (gap + 1.0)
    
    # Ensure non-finite values are handled
    H_valid = np.where(np.isfinite(H_valid), H_valid, 1e-9)
    
    H = np.where(valid, H_valid, 1e-9)
    
    # The diagonal (i==i) is never used for placement (j != i), but let's set it to 1e-9 to be clean
    np.fill_diagonal(H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
