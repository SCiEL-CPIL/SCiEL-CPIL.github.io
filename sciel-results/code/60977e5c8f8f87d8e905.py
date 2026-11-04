import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    # Generate candidate vectors with small squared norms
    # Norms squared: 2, 4, 6, 8
    candidates = []
    
    # Helper to add vector if not duplicate
    seen = set()
    def add_vec(v):
        key = tuple(v)
        if key not in seen:
            seen.add(key)
            candidates.append(np.array(v, dtype=np.int64))
            
    # Generate vectors with two +/- 1s (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                add_vec(v)
                
    # Generate vectors with one +/- 2 (norm^2 = 4)
    for i in range(D):
        for si in (1, -1):
            v = [0] * D
            v[i] = 2 * si
            add_vec(v)
            
    # Generate vectors with one +/- 2 and two +/- 1s (norm^2 = 6)
    for i in range(D):
        for si in (2, -2):
            for j, k in itertools.combinations([x for x in range(D) if x != i], 2):
                for sj in (1, -1):
                    for sk in (1, -1):
                        v = [0] * D
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        add_vec(v)
                        
    # Generate vectors with two +/- 2s (norm^2 = 8)
    for i, j in itertools.combinations(range(D), 2):
        for si in (2, -2):
            for sj in (2, -2):
                v = [0] * D
                v[i], v[j] = si, sj
                add_vec(v)

    # Greedy selection for a fixed max norm R
    # We try different R values and keep the best set
    best_set = []
    best_score = 0
    
    # Sort candidates by norm descending? 
    # If we pick large norms first, we set a high bar for distance.
    # If we pick small norms first, we might fill up with close vectors that block larger ones.
    # However, if we pick large norms, the distance requirement is high, so we might pick fewer.
    # Let's try sorting by norm descending.
    candidates.sort(key=lambda x: -np.dot(x, x))
    
    # Try R = sqrt(8) (norm^2 = 8)
    # And R = 2 (norm^2 = 4)
    # And R = sqrt(2) (norm^2 = 2)
    
    def try_greedy(max_norm_sq):
        selected = []
        max_norm = np.sqrt(max_norm_sq)
        
        for v in candidates:
            v_norm_sq = np.dot(v, v)
            if v_norm_sq > max_norm_sq:
                continue
                
            # Check if v can be added
            can_add = True
            for u in selected:
                diff = v - u
                dist_sq = np.dot(diff, diff)
                # Condition: dist >= max_norm
                # dist_sq >= max_norm_sq
                if dist_sq < max_norm_sq - 1e-9:
                    can_add = False
                    break
            
            if can_add:
                selected.append(v)
                
        return selected

    # We need to be careful: the max norm of the SELECTED set might be less than max_norm_sq
    # if we don't pick any vector with that norm.
    # But the condition is "min dist >= max norm of the set".
    # If we restrict to vectors with norm <= R, and enforce dist >= R, 
    # then dist >= R >= max_norm_of_set. So it's valid.
    
    # Try R^2 = 8
    set_8 = try_greedy(8)
    if len(set_8) > best_score:
        best_score = len(set_8)
        best_set = set_8
        
    if time.time() > deadline:
        return np.array(best_set, dtype=np.int64)
        
    # Try R^2 = 4
    set_4 = try_greedy(4)
    if len(set_4) > best_score:
        best_score = len(set_4)
        best_set = set_4
        
    if time.time() > deadline:
        return np.array(best_set, dtype=np.int64)
        
    # Try R^2 = 2
    set_2 = try_greedy(2)
    if len(set_2) > best_score:
        best_score = len(set_2)
        best_set = set_2

    return np.array(best_set, dtype=np.int64)
