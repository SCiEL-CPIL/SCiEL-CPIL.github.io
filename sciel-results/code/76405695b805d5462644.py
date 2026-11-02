"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    # 1. Generate weight-4 vectors
    w4_vectors = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(support, signs):
                v[idx] = val
            w4_vectors.append(v)
    w4_vectors = np.array(w4_vectors, dtype=np.int64)
    
    # 2. Generate weight-2 vectors
    w2_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                w2_vectors.append(v)
    w2_vectors = np.array(w2_vectors, dtype=np.int64)
    
    best_selected = []
    
    # Greedy search on weight-4 vectors
    for restart in range(100):
        if time.time() > deadline:
            break
            
        np.random.seed(restart)
        idxs = np.random.permutation(len(w4_vectors))
        
        current_selected = []
        
        for idx in idxs:
            if time.time() > deadline:
                break
                
            v = w4_vectors[idx]
            if not current_selected:
                current_selected.append(v)
                continue
                
            sel_arr = np.array(current_selected)
            dots = sel_arr @ v
            # Condition for w4-w4: dot <= 2
            if np.max(dots) <= 2:
                current_selected.append(v)
                
        if len(current_selected) > len(best_selected):
            best_selected = current_selected
            
    # 3. Try to add weight-2 vectors to best_selected
    if best_selected:
        best_arr = np.array(best_selected)
        # Condition for w2-w4: dot <= 1
        # Condition for w2-w2: dot <= 1 (since max norm is 2, dist^2 >= 4 => 2+2-2dot >= 4 => dot <= 0? 
        # Wait. If we add w2, max norm remains 2 (from w4).
        # So w2 must be dist >= 2 from w4 AND w2.
        # w2-w4: dot <= 1.
        # w2-w2: dist^2 >= 4 => 4 - 2dot >= 4 => dot <= 0.
        
        # So new w2 must have dot <= 1 with all w4s, and dot <= 0 with all w2s.
        
        added_w2 = []
        for v in w2_vectors:
            if time.time() > deadline:
                break
                
            # Check against existing best (w4s)
            dots_w4 = best_arr @ v
            if np.max(dots_w4) > 1:
                continue
                
            # Check against added w2s
            if added_w2:
                added_arr = np.array(added_w2)
                dots_w2 = added_arr @ v
                if np.max(dots_w2) > 0:
                    continue
            
            added_w2.append(v)
            
        if added_w2:
            best_selected.extend(added_w2)
            
    if not best_selected:
        # Fallback to seed if something went wrong
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return np.array(best_selected, dtype=np.int64)
# EVOLVE-BLOCK-END
