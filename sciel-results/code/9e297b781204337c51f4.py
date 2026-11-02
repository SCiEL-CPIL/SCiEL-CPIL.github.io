import itertools
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin

    D = 11
    
    # 1. Generate all roots of D11 lattice
    # Norm squared = 2
    roots = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    roots.append(v)
    roots_arr = np.array(roots, dtype=np.float64)
    n_roots = len(roots_arr)
    
    # 2. Generate Type B candidates (4 non-zero entries of +/- 1)
    # Norm squared = 4
    candidates = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(support, signs):
                v[pos] = sign
            candidates.append(v)
            
    cands = np.array(candidates, dtype=np.float64)
    n_cands = len(cands)
    
    # 3. Filter candidates compatible with ALL roots
    # Condition: root . cand <= 1
    # dots_roots_cands: (n_roots, n_cands)
    # Memory: 220 * 5280 * 8 bytes ~ 9.3 MB. OK.
    
    if time.time() < deadline - 20:
        dots_roots_cands = roots_arr @ cands.T
        max_dots = np.max(dots_roots_cands, axis=0)
        valid_cand_mask = max_dots <= 1
        valid_cands = cands[valid_cand_mask]
    else:
        valid_cands = np.array([], dtype=np.float64)
        
    # 4. Select a subset of valid_cands with pairwise dot product <= 2
    # We want to maximize the size of this subset.
    
    if len(valid_cands) == 0:
        return roots_arr.astype(np.int64)
        
    n_valid = len(valid_cands)
    
    # Precompute dot products between valid candidates?
    # n_valid might be large. Let's check.
    # If n_valid > 1000, full matrix is 1M floats (8MB). OK.
    # But checking all pairs is O(N^2).
    
    # Greedy selection with random restarts
    best_set = []
    best_count = 0
    
    # Heuristic: Sort by some property? Or just random.
    # Let's try a few random restarts of greedy selection.
    
    max_restarts = 10
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        rng = np.random.RandomState(r)
        indices = rng.permutation(n_valid)
        
        selected_indices = []
        selected_vectors = []
        # For speed, we can maintain a list of selected vectors and check against new ones.
        # But checking against all selected is O(K) per candidate.
        # Total O(N*K).
        
        for idx in indices:
            if time.time() > deadline - 5:
                break
                
            v = valid_cands[idx]
            
            # Check compatibility with already selected
            if len(selected_vectors) > 0:
                # Compute dot products with all selected
                # selected_vectors is a list of arrays.
                # Convert to array for batch dot?
                # Or just loop if K is small.
                # K will be maybe 100-200.
                # N is 5000.
                # 5000 * 200 = 1M dots. Fast.
                
                # To speed up, we can pre-compute a matrix of valid_cands
                # But we are selecting a subset.
                
                # Let's just check
                compatible = True
                for sv in selected_vectors:
                    if np.dot(v, sv) > 2:
                        compatible = False
                        break
                        
                if not compatible:
                    continue
            
            selected_indices.append(idx)
            selected_vectors.append(v)
            
        if len(selected_indices) > best_count:
            best_count = len(selected_indices)
            best_set = selected_indices[:]
            
    # Combine roots and best selected Type B vectors
    if best_count > 0:
        norm4_vectors = valid_cands[best_set]
        result = np.vstack([roots_arr, norm4_vectors]).astype(np.int64)
    else:
        result = roots_arr.astype(np.int64)
        
    return result
