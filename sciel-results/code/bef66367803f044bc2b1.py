"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _build_candidates():
    """Build candidate pool of nonzero integer vectors in R^11.
    
    We include:
    - Weight-1 vectors: 2*e_i (norm 2) -> 22 vectors
    - Weight-2 vectors: e_i +/- e_j (norm sqrt(2)) -> 440 vectors
    - Weight-3 vectors: e_i +/- e_j +/- e_k (norm sqrt(3)) -> 1320 vectors
    - Weight-4 vectors: e_i +/- e_j +/- e_k +/- e_l (norm 2) -> 3300 vectors
    """
    candidates = []
    
    # Weight-1: 2*e_i
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            candidates.append(v)
    
    # Weight-2: e_i +/- e_j
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Weight-3: e_i +/- e_j +/- e_k
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
    
    # Weight-4: e_i +/- e_j +/- e_k +/- e_l
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        candidates.append(v)
    
    return np.array(candidates, dtype=np.int64)


def construct():
    deadline = time.time() + 110  # safety margin before 150s kill
    
    candidates = _build_candidates()
    n = len(candidates)
    
    # Compute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Strategy: iterate over possible max norm squared M values.
    # For a given M, filter to vectors with norm_sq <= M,
    # build conflict graph where two nodes conflict if dist_sq < M,
    # and find max independent set using degree-ordered greedy.
    
    best_set_indices = []
    best_M = None
    
    # Possible M values from our candidate pool: 2, 3, 4
    possible_Ms = sorted(set(norms_sq.tolist()))
    
    for M in possible_Ms:
        if time.time() > deadline - 5:
            break
        
        mask = norms_sq <= M
        idx = np.where(mask)[0]
        sub = candidates[idx]
        m = len(sub)
        if m == 0:
            continue
        
        # Compute pairwise distances squared
        # dist^2[i,j] = ||sub[i] - sub[j]||^2 = ||sub[i]||^2 + ||sub[j]||^2 - 2*sub[i].sub[j]
        dot = sub @ sub.T
        ns = np.sum(sub * sub, axis=1)
        dist_sq = ns[:, None] + ns[None, :] - 2 * dot
        
        # Conflict: dist_sq[i,j] < M (and i != j)
        conflict = (dist_sq < M).astype(np.int8)
        np.fill_diagonal(conflict, 0)
        
        # Degree-ordered greedy max independent set
        degrees = conflict.sum(axis=1)
        order = np.argsort(degrees, kind='stable')
        
        selected = []
        remaining = set(range(m))
        
        for idx_pos in order:
            if idx_pos not in remaining:
                continue
            # Check if idx_pos is compatible with all already selected
            compatible = True
            for s in selected:
                if conflict[idx_pos, s] == 1:
                    compatible = False
                    break
            if compatible:
                selected.append(idx_pos)
                # Remove all neighbors of idx_pos from remaining
                neighbors = np.where(conflict[idx_pos] == 1)[0]
                for nb in neighbors:
                    remaining.discard(nb)
                remaining.discard(idx_pos)
        
        if len(selected) > len(best_set_indices):
            best_set_indices = selected
            best_M = M
    
    if not best_set_indices or best_M is None:
        # Fallback: return just weight-1 vectors (first 22)
        return candidates[:22]
    
    # Map back to original indices
    mask = norms_sq <= best_M
    idx = np.where(mask)[0]
    result = candidates[idx[best_set_indices]]
    
    # Verify the result satisfies the constraint
    if len(result) >= 2:
        max_norm_sq = int(np.max(np.sum(result * result, axis=1)))
        n_res = len(result)
        valid = True
        for i in range(n_res):
            if not valid:
                break
            diffs = result[i+1:] - result[i]
            dists_sq = np.sum(diffs * diffs, axis=1)
            if np.any(dists_sq < max_norm_sq):
                valid = False
        if not valid:
            # Fall back to weight-2 vectors only
            vecs = []
            for i, j in itertools.combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        vecs.append(v)
            result = np.array(vecs, dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
