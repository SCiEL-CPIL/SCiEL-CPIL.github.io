import numpy as np
import time

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 580  # safety margin
    
    # Generate all candidate vectors with entries in {-1,0,1} and weight 2 or 3
    candidates = []
    
    # Weight 2 vectors
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
    
    # Weight 3 vectors
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        for s3 in [-1, 1]:
                            v = np.zeros(d, dtype=np.int64)
                            v[i] = s1
                            v[j] = s2
                            v[k] = s3
                            candidates.append(v)
    
    n_cand = len(candidates)
    candidates_arr = np.array(candidates)
    
    best_set = None
    best_k = 0
    
    # Try multiple random orderings
    rng = np.random.default_rng(42)
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        order = rng.permutation(n_cand)
        selected = []
        max_norm_sq = 0
        
        for idx in order:
            if time.time() > deadline:
                break
            
            v = candidates_arr[idx]
            v_norm_sq = int(np.dot(v, v))
            new_max_norm = max(max_norm_sq, v_norm_sq)
            
            if len(selected) == 0:
                selected.append(v)
                max_norm_sq = new_max_norm
                continue
            
            # Check distance to all selected
            sel_arr = np.array(selected)
            diffs = sel_arr - v
            dists_sq = np.sum(diffs * diffs, axis=1)
            
            if np.min(dists_sq) >= new_max_norm:
                selected.append(v)
                max_norm_sq = new_max_norm
        
        if len(selected) > best_k:
            best_k = len(selected)
            best_set = selected
    
    if best_set is None or len(best_set) < 2:
        # Fallback: just weight-2 vectors
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        points.append(v)
        return np.array(points)
    
    return np.array(best_set)
