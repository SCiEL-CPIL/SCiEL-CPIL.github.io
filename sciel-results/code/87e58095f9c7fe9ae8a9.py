"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Combines a data-driven co-occurrence prior (from randomized BFD simulations)
    with a structural gap-utility heuristic.
    
    The co-occurrence matrix captures which items frequently appear together in
    high-quality packings. The gap-utility heuristic rewards pairs that leave
    small residual space (tight packing). We blend these two signals.
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Leave 50s margin for ACO and overhead
    
    n = len(demands)
    demands = np.asarray(demands, dtype=float)
    
    # Pre-compute basic structures
    pair_sum = demands[:, None] + demands[None, :]
    valid = pair_sum <= capacity
    gap = capacity - pair_sum
    
    # Build frequency and cumulative count for gap utility
    demand_ints = np.round(demands).astype(int)
    max_demand = int(np.max(demand_ints))
    
    freq = np.zeros(max_demand + 1, dtype=np.int64)
    for d in demand_ints:
        freq[d] += 1
    cumsum = np.cumsum(freq)
    
    # Compute gap-utility heuristic (structural)
    H_struct = np.full((n, n), 1e-9)
    H_struct[valid] = 0.0
    
    if np.any(valid):
        valid_indices = np.where(valid)
        i_idx = valid_indices[0]
        j_idx = valid_indices[1]
        gap_valid = gap[valid]
        
        gap_clipped = np.clip(gap_valid, 0, max_demand).astype(int)
        fit_counts = cumsum[gap_clipped]
        
        i_fits = (2.0 * demands[i_idx] + demands[j_idx]) <= capacity
        j_fits = (demands[i_idx] + 2.0 * demands[j_idx]) <= capacity
        
        adjusted_counts = fit_counts - i_fits.astype(np.int64) - j_fits.astype(np.int64)
        adjusted_counts = np.maximum(adjusted_counts, 0)
        
        # Reward pairs with fewer fitting items in residual (tighter packing)
        # Use a stronger penalty for larger gaps to emphasize tight packing
        gap_penalty = 1.0 / (1.0 + 0.1 * gap_valid)
        H_struct[valid] = gap_penalty * (1.0 / (adjusted_counts + 1.0))
        
        # Boost exact fits
        exact_fit = valid & (gap == 0)
        H_struct[exact_fit] = 10.0
    
    # Compute co-occurrence prior (data-driven) using BFD simulations
    C = np.zeros((n, n), dtype=np.int32)
    bin_counts = np.zeros(n, dtype=np.int32)  # Track how many bins each item ends up in
    
    # Run randomized BFD simulations
    sorted_indices = np.argsort(-demands)
    
    num_sims = 500  # Start with a reasonable number
    sim_count = 0
    
    for sim in range(num_sims):
        if time.time() > deadline:
            break
            
        # Create a shuffled version where ties are broken randomly
        noise = np.random.rand(n)
        sort_keys = np.lexsort((noise, -demands))
        
        # BFD Simulation: place each item in the bin where it fits with smallest remaining space
        bins = []
        bin_loads = []
        
        for idx in sort_keys:
            d = demands[idx]
            best_bin = -1
            best_remaining = float('inf')
            
            for b in range(len(bins)):
                remaining = capacity - bin_loads[b]
                if remaining >= d and remaining < best_remaining:
                    best_remaining = remaining
                    best_bin = b
            
            if best_bin >= 0:
                bins[best_bin].append(idx)
                bin_loads[best_bin] += d
            else:
                bins.append([idx])
                bin_loads.append(d)
        
        # Record co-occurrences with quality weighting
        for b in bins:
            if len(b) > 1:
                # Weight co-occurrences by how full the bin is (tighter = better)
                bin_fill = sum(demands[x] for x in b) / capacity
                weight = 1.0 + 2.0 * bin_fill  # Range [1, 3]
                
                for i in range(len(b)):
                    for j in range(i + 1, len(b)):
                        u = b[i]
                        v = b[j]
                        C[u, v] += int(weight)
                        C[v, u] += int(weight)
        
        sim_count += 1
        
        # If we have time, run more simulations
        if time.time() < deadline - 10.0 and sim_count < 2000:
            num_sims = min(num_sims + 100, 2000)
    
    # Transform co-occurrence counts to heuristic scores
    max_c = np.max(C) if sim_count > 0 else 0
    
    if max_c > 0:
        # Normalize and scale co-occurrence scores
        C_norm = C.astype(float) / max_c
        # Use a power law to amplify frequent co-occurrences
        H_coco = 1.0 + 10.0 * (C_norm ** 0.7)
        H_coco[C == 0] = 0.0  # No co-occurrence gets no boost
    else:
        H_coco = np.zeros((n, n), dtype=float)
    
    # Blend the two heuristics
    # We want to strongly encourage pairs that both fit well structurally
    # AND are frequently co-occurring in good solutions.
    # Use a weighted combination with emphasis on co-occurrence when available.
    
    H = np.full((n, n), 1e-9)
    H[valid] = 0.0
    
    if np.any(valid):
        # For valid pairs, combine structural and co-occurrence signals
        struct_valid = H_struct[valid]
        coco_valid = H_coco[valid]
        
        # If co-occurrence data is available, weight it heavily
        if max_c > 0:
            # Blend: 0.35 * structural + 0.65 * co-occurrence
            H[valid] = 0.35 * struct_valid + 0.65 * coco_valid
        else:
            # Fallback to structural only
            H[valid] = struct_valid
        
        # Ensure exact fits still get a strong boost
        exact_fit = valid & (gap == 0)
        H[exact_fit] = np.maximum(H[exact_fit], 10.0)
    
    # Ensure non-positive values become 1e-9
    H[H <= 0] = 1e-9
    
    # Ensure diagonal is low (items don't pair with themselves)
    np.fill_diagonal(H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
