"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing algorithm.
    
    Combines a structural gap-utility heuristic with a data-driven co-occurrence prior
    from randomized BFD (Best-Fit-Decreasing) simulations.
    
    The structural component rewards pairs that leave small residual space (tight packing)
    by penalizing pairs where many other items could still fit in the remaining space.
    
    The co-occurrence component captures which items frequently appear together in
    high-quality packings found via BFD simulations.
    
    We blend these two signals to guide the ACO toward tight, high-quality packings.
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Leave 50s margin for ACO and overhead
    
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
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
    
    # Compute structural gap-utility heuristic
    H_struct = np.full((n, n), 1e-9)
    H_struct[valid] = 0.0
    
    if np.any(valid):
        valid_indices = np.where(valid)
        i_idx = valid_indices[0]
        j_idx = valid_indices[1]
        gap_valid = gap[valid]
        
        # Count items that could fit in the remaining space
        gap_clipped = np.clip(gap_valid, 0, max_demand).astype(int)
        fit_counts = cumsum[gap_clipped]
        
        # Exclude items i and j themselves from the count
        i_fits = (2.0 * demands[i_idx] + demands[j_idx]) <= capacity
        j_fits = (demands[i_idx] + 2.0 * demands[j_idx]) <= capacity
        
        adjusted_counts = fit_counts - i_fits.astype(np.int64) - j_fits.astype(np.int64)
        adjusted_counts = np.maximum(adjusted_counts, 0)
        
        # Reward pairs with fewer fitting items in residual (tighter packing)
        # Use a combination of gap penalty and count penalty
        gap_penalty = 1.0 / (1.0 + 0.1 * gap_valid)
        count_penalty = 1.0 / (1.0 + adjusted_counts)
        H_struct[valid] = gap_penalty * count_penalty
        
        # Boost exact fits strongly
        exact_fit = valid & (gap == 0)
        H_struct[exact_fit] = 10.0
    
    # Compute co-occurrence prior (data-driven) using BFD simulations
    C = np.zeros((n, n), dtype=np.int32)
    
    # Run randomized BFD simulations
    sorted_indices = np.argsort(-demands)
    
    num_sims = 300  # Start with a reasonable number
    sim_count = 0
    
    for sim in range(num_sims):
        if time.time() > deadline:
            break
        
        # Create a shuffled version where ties are broken randomly
        noise = np.random.rand(n)
        sort_keys = np.lexsort((noise, -demands))
        
        # BFD Simulation
        bins = []
        bin_loads = []
        
        for idx in sort_keys:
            d = demands[idx]
            best_bin = -1
            best_fit = capacity + 1  # We want to minimize remaining space
            
            for b in range(len(bins)):
                remaining = capacity - bin_loads[b]
                if d <= remaining:
                    # BFD: choose the bin with the smallest remaining space that fits
                    if remaining < best_fit:
                        best_fit = remaining
                        best_bin = b
            
            if best_bin != -1:
                bins[best_bin].append(idx)
                bin_loads[best_bin] += d
            else:
                bins.append([idx])
                bin_loads.append(d)
        
        # Record co-occurrences
        for b in bins:
            if len(b) > 1:
                for i in range(len(b)):
                    for j in range(i + 1, len(b)):
                        u = b[i]
                        v = b[j]
                        C[u, v] += 1
                        C[v, u] += 1
        
        sim_count += 1
        
        # If we have time, run more simulations
        if time.time() < deadline - 10.0 and sim_count < 1000:
            num_sims = min(num_sims + 50, 1000)
    
    # Transform co-occurrence counts to heuristic scores
    max_c = np.max(C) if sim_count > 0 else 0
    
    if max_c > 0:
        # Normalize and scale co-occurrence scores
        C_norm = C.astype(float) / max_c
        # Use a power law to amplify frequent co-occurrences
        H_coco = 1.0 + 10.0 * (C_norm ** 0.5)
        H_coco[C == 0] = 0.0  # No co-occurrence gets no boost
    else:
        H_coco = np.zeros((n, n), dtype=float)
    
    # Blend the two heuristics
    H = np.full((n, n), 1e-9)
    H[valid] = 0.0
    
    if np.any(valid):
        struct_valid = H_struct[valid]
        coco_valid = H_coco[valid]
        
        # Blend: 0.4 * structural + 0.6 * co-occurrence
        # Co-occurrence is data-driven and should be weighted more heavily
        if max_c > 0:
            H[valid] = 0.4 * struct_valid + 0.6 * coco_valid
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
