import itertools
import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Generate all weight-4 vectors (squared norm 4)
    # These are vectors with exactly four non-zero entries, each +1 or -1.
    # Norm is 2.
    # Constraint: min pairwise distance >= max norm (2).
    # dist^2 >= 4.
    # |u-v|^2 = 4 + 4 - 2 u.v = 8 - 2 u.v.
    # 8 - 2 u.v >= 4 => 4 >= 2 u.v => u.v <= 2.
    
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates)
    
    # Precompute dot products
    dot_matrix = candidates @ candidates.T
    
    # Conflict if dot > 2
    conflict_mask = dot_matrix > 2
    np.fill_diagonal(conflict_mask, False)
    conflict_degrees = np.sum(conflict_mask, axis=1)
    
    best_set = []
    rng = np.random.default_rng(42)
    
    for restart in range(100):
        if time.time() > deadline - 5.0:
            break
            
        if restart == 0:
            order = np.argsort(conflict_degrees)
        else:
            order = rng.permutation(n_candidates)
            
        selected_indices = []
        available = np.ones(n_candidates, dtype=bool)
        
        for idx in order:
            if not available[idx]:
                continue
                
            selected_indices.append(idx)
            available[conflict_mask[idx]] = False
            available[idx] = False
            
        current_set = candidates[selected_indices]
        
        if len(current_set) > len(best_set):
            best_set = current_set
            
    if not best_set:
        # Fallback to known valid set of 242 vectors (N=1 and N=2)
        vecs = []
        for i in range(D):
            for s in [1, -1]:
                v = np.zeros(D, dtype=np.int64)
                v[i] = s
                vecs.append(v)
        for i in range(D):
            for j in range(i+1, D):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        vecs.append(v)
        best_set = np.array(vecs, dtype=np.int64)
        
    return best_set
