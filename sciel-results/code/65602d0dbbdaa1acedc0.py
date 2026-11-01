"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate candidate integer vectors with small entries
    # Focus on vectors with norm squared in {1, 2, 3, 4}
    # Entries in {-1, 0, 1} primarily, maybe some with 2
    
    candidates = []
    
    # Vectors with exactly 1 nonzero entry (norm^2 = 1 or 4)
    for i in range(D):
        for val in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = val
            candidates.append(v)
    
    # Vectors with exactly 2 nonzero entries (norm^2 = 2, 5, 8)
    for i in range(D):
        for j in range(i+1, D):
            for vi in (1, -1):
                for vj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j] = vi, vj
                    candidates.append(v)
    
    # Vectors with exactly 3 nonzero entries (norm^2 = 3, 6, 9, etc.)
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for vi in (1, -1):
                    for vj in (1, -1):
                        for vk in (1, -1):
                            v = np.zeros(D, dtype=np.int64)
                            v[i], v[j], v[k] = vi, vj, vk
                            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    
    # Compute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # We want to select a subset where:
    # min_{i!=j} ||v_i - v_j||^2 >= max_k ||v_k||^2
    # i.e., for all i != j: ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j >= max_norm_sq
    # where max_norm_sq = max of selected norms
    
    # Strategy: try different max_norm_sq values and find max independent set
    # on conflict graph where edge (i,j) exists if constraint violated
    
    best_set = None
    best_size = 0
    
    # Try with different norm thresholds
    for max_norm_sq in sorted(set(norms_sq)):
        if time.time() > deadline - 5.0:
            break
        
        # Filter candidates with norm_sq <= max_norm_sq
        mask = norms_sq <= max_norm_sq
        idx = np.where(mask)[0]
        
        if len(idx) < best_size:
            continue
        
        sub_candidates = candidates[idx]
        sub_norms = norms_sq[idx]
        
        # Build conflict matrix: conflict if ||v_i - v_j||^2 < max_norm_sq
        # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j
        # We need this >= max_norm_sq for all pairs in selected set
        
        # Compute dot products
        dots = sub_candidates @ sub_candidates.T
        norms_sq_2d = np.outer(sub_norms, np.ones(len(sub_norms))) + np.outer(np.ones(len(sub_norms)), sub_norms)
        dist_sq = norms_sq_2d - 2 * dots
        
        # Conflict if dist_sq < max_norm_sq (and i != j)
        conflict = (dist_sq < max_norm_sq).astype(np.int8)
        np.fill_diagonal(conflict, 0)
        
        # Greedy maximal independent set with random restarts
        n = len(sub_candidates)
        
        def greedy_iset(conflict_mat, seed, time_limit):
            rng = np.random.default_rng(seed)
            start = time.time()
            n_local = len(conflict_mat)
            selected = []
            available = set(range(n_local))
            
            # Random order
            order = rng.permutation(n_local)
            
            for i in order:
                if time.time() > time_limit:
                    break
                if i not in available:
                    continue
                # Check if i conflicts with any already selected
                ok = True
                for j in selected:
                    if conflict_mat[i, j]:
                        ok = False
                        break
                if ok:
                    selected.append(i)
                    # Remove all neighbors of i from available
                    for j in available:
                        if j != i and conflict_mat[i, j]:
                            available.discard(j)
                    available.discard(i)
            
            return selected
        
        # Try a few greedy passes
        for attempt in range(3):
            if time.time() > deadline - 5.0:
                break
            selected = greedy_iset(conflict, attempt * 1000 + max_norm_sq, deadline - 10.0)
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = [idx[s] for s in selected]
    
    if best_set is None:
        # Fallback: return the two-nonzero vectors (known valid)
        vectors = []
        for i in range(D):
            for j in range(i+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j] = si, sj
                        vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return candidates[best_set]

# EVOLVE-BLOCK-END
