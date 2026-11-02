import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Generate candidates
    candidates = []
    
    # Type 1: Norm 2, One 2, rest 0
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Norm 2, Four 1s, rest 0
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    # Type 3: Norm sqrt(8), Two 2s, rest 0
    for i, j in itertools.combinations(range(D), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                candidates.append(v)
                
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    def greedy_search(allowed_mask, r_sq, orderings):
        best_set = []
        for order in orderings:
            if time.time() > deadline:
                break
            selected = []
            selected_mask = np.zeros(N, dtype=bool)
            
            for idx in order:
                if not allowed_mask[idx]:
                    continue
                    
                if not selected:
                    selected.append(idx)
                    selected_mask[idx] = True
                    continue
                    
                # Check compatibility
                dots = M[idx, selected_mask]
                sel_norms = norms_sq[selected_mask]
                thresholds = (norms_sq[idx] + sel_norms - r_sq) / 2.0
                
                if np.all(dots <= thresholds):
                    selected.append(idx)
                    selected_mask[idx] = True
                    
            if len(selected) > len(best_set):
                best_set = selected
                
        return best_set

    # Masks for allowed candidates
    norm4_mask = (norms_sq == 4)
    all_mask = np.ones(N, dtype=bool)
    
    # Generate random orderings
    rng = np.random.default_rng(42)
    orderings = [np.arange(N)]
    for _ in range(100):
        orderings.append(rng.permutation(N))
        
    # Case 1: Only norm 4 vectors, R^2 = 4
    best_set_4 = greedy_search(norm4_mask, 4, orderings)
    
    # Case 2: All vectors (Norm 4 and 8), R^2 = 8
    # Note: This imposes dot <= 0 on Norm 4 pairs, which is stricter.
    # But allows Norm 8 vectors.
    best_set_8 = greedy_search(all_mask, 8, orderings)
    
    # Case 3: Only norm 8 vectors, R^2 = 8
    norm8_mask = (norms_sq == 8)
    best_set_8_only = greedy_search(norm8_mask, 8, orderings)
    
    # Select the best
    best_set_indices = max([best_set_4, best_set_8, best_set_8_only], key=len)
    
    if not best_set_indices:
        # Fallback
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return candidates[best_set_indices].astype(np.int64)
