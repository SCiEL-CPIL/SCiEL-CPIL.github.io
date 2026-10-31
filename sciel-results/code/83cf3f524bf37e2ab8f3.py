"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Generate candidate vectors with entries in {-1, 0, 1}
    # Start with weight 2, then weight 3, etc.
    candidates = []
    
    # Weight 2 vectors: C(11,2) * 4 = 55 * 4 = 220
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Weight 3 vectors: C(11,3) * 8 = 165 * 8 = 1320
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
    
    # Sort by norm (prefer lower norm first to keep max norm small)
    norms = np.array([np.dot(v, v) for v in candidates])
    order = np.argsort(norms)
    candidates = [candidates[i] for i in order]
    
    selected = []
    selected_norms = []
    
    for v in candidates:
        if time.time() > deadline:
            break
        
        v_norm2 = int(np.dot(v, v))
        if v_norm2 == 0:
            continue
        
        # Check constraint: for all selected u, ||v - u||^2 >= max(v_norm2, max_selected_norm2)
        # Actually the constraint is: min pairwise distance >= max norm
        # So for the new vector v, we need ||v - u|| >= max_norm for all u in selected
        # where max_norm = max(v_norm, all selected norms)
        
        if not selected:
            selected.append(v)
            selected_norms.append(v_norm2)
            continue
        
        current_max_norm2 = max(v_norm2, max(selected_norms))
        
        # Check against all selected vectors
        ok = True
        for u in selected:
            diff = v - u
            dist2 = int(np.dot(diff, diff))
            if dist2 < current_max_norm2:
                ok = False
                break
        
        if ok:
            selected.append(v)
            selected_norms.append(v_norm2)
    
    if not selected:
        # Fallback: return the seed solution
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return np.array(selected, dtype=np.int64)
# EVOLVE-BLOCK-END
