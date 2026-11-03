import numpy as np

def construct():
    D = 11
    candidates = []
    
    # Generate all integer vectors with norm squared <= 4
    # This covers a dense set of small vectors which are likely to satisfy
    # the distance constraint relative to their norm.
    
    # Norm squared 1: (1, 0, ..., 0) and (-1, 0, ..., 0)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = s
            candidates.append(v)
            
    # Norm squared 2: (1, 1, 0, ...) and sign variants
    for i in range(D):
        for j in range(i + 1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
                    
    # Norm squared 3: (1, 1, 1, 0, ...) and sign variants
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        for sk in [1, -1]:
                            v = [0] * D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
                            
    # Norm squared 4:
    # Type A: (2, 0, ...)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = s * 2
            candidates.append(v)
            
    # Type B: (1, 1, 1, 1, 0, ...)
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for l in range(k + 1, D):
                    for si in [1, -1]:
                        for sj in [1, -1]:
                            for sk in [1, -1]:
                                for sl in [1, -1]:
                                    v = [0] * D
                                    v[i] = si
                                    v[j] = sj
                                    v[k] = sk
                                    v[l] = sl
                                    candidates.append(v)
                                    
    # Sort candidates by norm squared (ascending) to greedily add smaller norms first
    candidates = np.array(candidates, dtype=np.int64)
    norms_sq = np.sum(candidates**2, axis=1)
    sorted_indices = np.argsort(norms_sq)
    sorted_candidates = candidates[sorted_indices]
    
    selected = []
    
    for idx in range(len(sorted_candidates)):
        v = sorted_candidates[idx]
        norm_v_sq = np.sum(v**2)
        
        if not selected:
            selected.append(v)
            continue
            
        # Check distance to all selected vectors
        # Vectorized check
        sel_arr = np.array(selected)
        diffs = sel_arr - v
        dists_sq = np.sum(diffs**2, axis=1)
        min_dist_sq = np.min(dists_sq)
        
        # Constraint: min_dist >= max_norm
        # Since we add in ascending norm order, max_norm is sqrt(norm_v_sq)
        # So we need min_dist_sq >= norm_v_sq
        
        if min_dist_sq >= norm_v_sq:
            selected.append(v)
            
    return np.array(selected, dtype=np.int64)
