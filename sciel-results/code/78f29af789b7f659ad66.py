import itertools
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    D = 11
    # Generate all weight 4 vectors
    vectors = []
    supports = []
    for indices in itertools.combinations(range(D), 4):
        s = frozenset(indices)
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sig in zip(indices, signs):
                v[idx] = sig
            vectors.append(v)
            supports.append(s)
            
    # Greedy Independent Set
    # Try a few random orders
    best_set = []
    
    for restart in range(5):
        if time.time() > deadline:
            break
            
        # Shuffle indices
        indices = list(range(len(vectors)))
        np.random.shuffle(indices)
        
        selected_idx = []
        selected_vecs = []
        
        for i in indices:
            if time.time() > deadline:
                break
            v = vectors[i]
            s = supports[i]
            conflict = False
            for j in selected_idx:
                u = vectors[j]
                su = supports[j]
                if s == su:
                    if v == u:
                        conflict = True
                        break
                    continue
                # Check intersection
                inter = s.intersection(su)
                if len(inter) >= 3:
                    # Check signs on intersection
                    ok = True
                    for idx in inter:
                        if v[idx] != u[idx]:
                            ok = False
                            break
                    if ok:
                        conflict = True
                        break
            if not conflict:
                selected_idx.append(i)
                selected_vecs.append(v)
                
        if len(selected_vecs) > len(best_set):
            best_set = selected_vecs
            
    if not best_set:
        # Fallback
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return np.array(best_set, dtype=np.int64)
