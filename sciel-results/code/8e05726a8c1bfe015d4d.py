"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time
from collections import defaultdict


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Strategy: Generate a large pool of candidate integer vectors,
    # then use greedy maximum independent set to find the largest
    # subset where min pairwise distance >= max norm.
    
    # Step 1: Generate candidate vectors
    # Use vectors with support 1-4, values in {-2,-1,0,1,2}
    # This gives a large diverse pool
    
    candidates = []
    
    # Support 1: single non-zero coordinate
    for coord in range(D):
        for val in [1, -1, 2, -2]:
            v = [0] * D
            v[coord] = val
            candidates.append(tuple(v))
    
    # Support 2: two non-zero coordinates
    for i, j in itertools.combinations(range(D), 2):
        for vi in [1, -1, 2, -2]:
            for vj in [1, -1, 2, -2]:
                v = [0] * D
                v[i] = vi
                v[j] = vj
                candidates.append(tuple(v))
    
    # Support 3: three non-zero coordinates with small values
    for i, j, k in itertools.combinations(range(D), 3):
        for vi in [1, -1]:
            for vj in [1, -1]:
                for vk in [1, -1]:
                    v = [0] * D
                    v[i] = vi
                    v[j] = vj
                    v[k] = vk
                    candidates.append(tuple(v))
        # Also try (2,1,0) type patterns
        for vi in [1, -1, 2, -2]:
            for vj in [1, -1]:
                v = [0] * D
                v[i] = vi
                v[j] = vj
                v[k] = 0
                candidates.append(tuple(v))
    
    # Support 4: four non-zero coordinates with +/-1
    for i, j, k, l in itertools.combinations(range(D), 4):
        for vi in [1, -1]:
            for vj in [1, -1]:
                for vk in [1, -1]:
                    for vl in [1, -1]:
                        v = [0] * D
                        v[i] = vi
                        v[j] = vj
                        v[k] = vk
                        v[l] = vl
                        candidates.append(tuple(v))
    
    # Remove duplicates
    candidates = list(set(candidates))
    
    # Convert to numpy for faster computation
    cand_arr = np.array(candidates, dtype=np.int64)
    n_cand = len(cand_arr)
    
    # Compute norms squared for all candidates
    norms_sq = np.sum(cand_arr * cand_arr, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Filter: only keep vectors with norm <= some threshold
    # We want vectors whose norm is close to max, so distance constraint is tight
    # Actually, for kissing number, we want all vectors to have the SAME norm ideally
    # Let's group by norm_sq and work with the most common norm
    
    # Find the norm that appears most frequently among "good" candidates
    # Or better: use vectors with norm_sq = 4 (support 2 with +/-2, or support 1 with +/-2)
    # or norm_sq = 8 (various combinations)
    
    # Let's try multiple norm levels and pick the best result
    best_result = np.empty((0, D), dtype=np.int64)
    
    for target_norm_sq in [4, 5, 8, 9, 10]:
        if time.time() > deadline - 5.0:
            break
            
        # Get candidates with this exact norm
        mask = (norms_sq == target_norm_sq)
        if not np.any(mask):
            continue
            
        subset = cand_arr[mask]
        n_sub = len(subset)
        
        if n_sub < 2:
            continue
        
        # Compute pairwise distances squared
        # dist^2(i,j) = ||vi||^2 + ||vj||^2 - 2*vi.vj = 2*target_norm_sq - 2*vi.vj
        # So dist^2 >= target_norm_sq iff 2*target_norm_sq - 2*vi.vj >= target_norm_sq
        # iff target_norm_sq >= 2*vi.vj
        # iff vi.vj <= target_norm_sq/2
        
        # Compute all pairwise dot products
        dots = subset @ subset.T
        np.fill_diagonal(dots, 0)
        
        # Constraint: dist^2 >= target_norm_sq means 2*target_norm_sq - 2*dots >= target_norm_sq
        # i.e., target_norm_sq >= 2*dots, i.e., dots <= target_norm_sq/2
        threshold = target_norm_sq / 2.0
        
        # Build adjacency: i and j are "compatible" if dots[i,j] <= threshold
        # We want maximum clique in the compatibility graph (or max independent set in incompatibility)
        
        # For greedy: pick vectors one by one, maintaining constraint with all already picked
        selected = []
        remaining = list(range(n_sub))
        
        # Sort candidates by some heuristic (e.g., number of conflicts)
        # Fewer conflicts = more compatible = pick first
        conflict_counts = np.sum(dots > threshold, axis=1)
        order = np.argsort(conflict_counts)
        
        for idx in order:
            if time.time() > deadline - 3.0:
                break
            # Check if this candidate is compatible with all already selected
            if len(selected) == 0:
                selected.append(idx)
            else:
                sel_arr = np.array(selected)
                # Check dots[idx, sel_arr] <= threshold for all
                if np.all(dots[idx, sel_arr] <= threshold):
                    selected.append(idx)
        
        if len(selected) > len(best_result):
            best_result = subset[selected]
        
        if time.time() > deadline - 2.0:
            break
    
    # If best_result is empty, fall back to the original E8-based construction
    if len(best_result) == 0:
        vectors = []
        for i, j in itertools.combinations(range(8), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)
        for mask in range(256):
            if bin(mask).count("1") % 2 == 0:
                v = [0] * D
                for k in range(8):
                    if mask & (1 << k):
                        v[k] = -1
                    else:
                        v[k] = 1
                vectors.append(v)
        idxs = [8, 9, 10]
        for i, j in itertools.combinations(idxs, 2):
            k = idxs[0] + idxs[1] + idxs[2] - i - j
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    v[k] = 0
                    vectors.append(v)
        best_result = np.array(vectors, dtype=np.int64)
    
    return best_result
# EVOLVE-BLOCK-END
