"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    # Generate candidate vectors: all vectors with 1 or 2 nonzero entries, values in {-1, 1}
    candidates = []
    
    # Unit vectors: ±e_i
    for i in range(D):
        v = [0] * D
        v[i] = 1
        candidates.append(v)
        v[i] = -1
        candidates.append(v)
    
    # Two-nonzero vectors: ±e_i ± e_j
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.float64)
    norms = np.linalg.norm(candidates, axis=1)
    
    # Greedy: select vectors such that for all pairs, normalized dot <= 0
    # i.e., dot(v, w) <= 0 in original space (since norms are positive)
    selected = []
    selected_dots_ok = True
    
    # Sort candidates by norm descending to prioritize larger vectors? 
    # Actually for the dot<=0 constraint, norm doesn't matter for the sign.
    # Let's just go in order.
    
    for idx in range(len(candidates)):
        if time.time() > deadline - 5:
            break
        v = candidates[idx]
        # Check against all selected
        ok = True
        for w in selected:
            if np.dot(v, w) > 1e-9:
                ok = False
                break
        if ok:
            selected.append(v)
    
    if not selected:
        # Fallback: just unit vectors
        selected = [[0]*D for _ in range(2*D)]
        for i in range(D):
            selected[2*i][i] = 1
            selected[2*i+1][i] = -1
        return np.array(selected, dtype=np.int64)
    
    return np.array(selected, dtype=np.int64)
# EVOLVE-BLOCK-END
