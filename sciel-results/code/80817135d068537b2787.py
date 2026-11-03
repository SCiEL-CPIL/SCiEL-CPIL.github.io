"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Generate candidate integer vectors
    candidates = []
    
    # Vectors with two nonzero ±1 entries (norm^2 = 2)
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.append(v)
    
    # Vectors with one nonzero ±2 entry (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
    
    # Vectors with four nonzero ±1 entries (norm^2 = 4)
    for indices in combinations(range(D), 4):
        for signs in product((1, -1), repeat=4):
            v = [0] * D
            for idx, sign in zip(indices, signs):
                v[idx] = sign
            candidates.append(v)
    
    # Deduplicate
    unique_candidates = []
    seen = set()
    for v in candidates:
        key = tuple(v)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(np.array(v, dtype=np.int64))
    
    candidates = np.array(unique_candidates, dtype=np.int64)
    n = len(candidates)
    
    # Compute norms squared
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Compute all pairwise dot products
    dots = candidates @ candidates.T
    
    # Conflict matrix
    min_norms = np.minimum(norms_sq[:, None], norms_sq[None, :])
    conflict = (2 * dots) > min_norms
    np.fill_diagonal(conflict, False)
    
    # Greedy MIS
    selected = []
    remaining_mask = np.ones(n, dtype=bool)
    conflict_count = np.sum(conflict, axis=1).copy()
    
    while remaining_mask.any() and (time.time() - start_time) < deadline:
        # Find candidate with min conflict count among remaining
        masked_conflict = np.where(remaining_mask, conflict_count, np.inf)
        best_idx = np.argmin(masked_conflict)
        
        if not remaining_mask[best_idx]:
            break
        
        # Check compatibility with all selected
        if selected:
            selected_arr = np.array(selected)
            if np.any(conflict[best_idx, selected_arr]):
                remaining_mask[best_idx] = False
                continue
        
        # Add to selected
        selected.append(best_idx)
        remaining_mask[best_idx] = False
        
        # Update conflict counts
        conflict_count -= conflict[best_idx]
    
    if not selected:
        vectors = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    result = candidates[selected]
    return result
# EVOLVE-BLOCK-END
