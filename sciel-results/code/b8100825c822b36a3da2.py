# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    d = np.asarray(demands, dtype=float)
    n = len(d)

    # 1. Run Best-Fit-Decreasing (BFD) to get a reference packing
    order = np.argsort(-d, kind="stable")
    bin_loads = []
    item_bin = np.full(n, -1, dtype=int)
    for idx in order:
        size = d[idx]
        best_bi = -1
        best_rem = np.inf
        for bi in range(len(bin_loads)):
            rem = capacity - bin_loads[bi]
            if rem >= size and rem < best_rem:
                best_rem = rem
                best_bi = bi
        if best_bi >= 0:
            bin_loads[best_bi] += size
            item_bin[idx] = best_bi
        else:
            bin_loads.append(size)
            item_bin[idx] = len(bin_loads) - 1

    min_size = float(d.min())
    H = np.full((n, n), 1e-9, dtype=float)

    # 2. Compute heuristic based on BFD bin membership and fit quality
    # We iterate over bins in the BFD solution.
    # For each pair (i, j) in the same bin, we calculate a high score.
    # This guides the ACO to replicate the BFD solution.
    
    # Pre-calculate scores for all valid pairs to allow vectorization or efficient lookup?
    # n=120, so n^2 = 14400. A double loop is fine.
    
    for i in range(n):
        si = d[i]
        bi_i = item_bin[i]
        
        for j in range(n):
            if i == j:
                continue
            sj = d[j]
            combined = si + sj
            if combined > capacity + 1e-12:
                continue
            
            # Base score: Best Fit intuition (lower residual is better)
            residual = capacity - combined
            # Normalize residual: 0 is perfect, capacity is worst
            # Use a decay function
            slack_score = 1.0 / (1.0 + residual / 10.0)
            
            # Structural score: Is residual close to a multiple of min_size?
            # This helps if we can fill the rest with small items.
            if min_size > 0:
                k = int(round(residual / min_size))
                k = max(0, k)
                dist_to_k = abs(residual - k * min_size)
                dist_to_k1 = abs(residual - (k + 1) * min_size)
                min_dist = min(dist_to_k, dist_to_k1)
                # If min_dist is small, the residual is "structurally good"
                struct_score = 1.0 / (1.0 + min_dist / max(min_size, 1e-9))
            else:
                struct_score = 1.0

            base_score = slack_score * struct_score

            # Structural Change: Boost score if i and j are in the same BFD bin
            # This creates "attractor" basins around the BFD solution.
            bi_j = item_bin[j]
            if bi_i == bi_j:
                # Same bin in BFD: Strong attraction
                # Also consider how full that bin is. Fuller bins are "tighter" and more valuable.
                # Actually, just boosting is enough to prioritize BFD structure.
                H[i, j] = base_score * 5.0 
            else:
                # Different bins: Weak attraction, allowing exploration but biasing towards BFD structure
                H[i, j] = base_score * 0.5

    # Ensure finite and positive floor.
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    return H
# EVOLVE-BLOCK-END
