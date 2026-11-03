import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11
    
    # Generate all weight-4 supports
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)
    
    # Build incompatibility graph for supports
    # Two supports are incompatible if their intersection size is >= 3
    # Because if |S_i ∩ S_j| >= 3, there exist sign assignments such that u.v > 2.
    # Actually, we want to pick a set of supports such that ANY vectors from them are compatible.
    # If |S_i ∩ S_j| <= 2, then max dot product is 2, so all pairs are compatible.
    # If |S_i ∩ S_j| >= 3, we cannot include all sign variations of both.
    # So we restrict to supports with intersection <= 2.
    
    # Precompute intersection sizes
    # supports is a list of tuples
    # Create a set representation for fast intersection? Or just use sets.
    sup_sets = [set(s) for s in supports]
    
    # Incompatibility matrix for supports
    # incompat_sup[i, j] = 1 if |sup_sets[i] ∩ sup_sets[j]| >= 3
    incompat_sup = np.zeros((N_sup, N_sup), dtype=np.int8)
    
    for i in range(N_sup):
        if time.time() > deadline - 10.0:
            break
        for j in range(i + 1, N_sup):
            inter_size = len(sup_sets[i] & sup_sets[j])
            if inter_size >= 3:
                incompat_sup[i, j] = 1
                incompat_sup[j, i] = 1
    
    # Find Maximum Independent Set on this graph
    # Use greedy with random restarts
    rng = np.random.default_rng(42)
    best_iset = None
    best_size = 0
    
    n_restarts = 100
    for r in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Greedy: random order, pick if compatible
        order = rng.permutation(N_sup)
        selected = np.zeros(N_sup, dtype=bool)
        
        for idx in order:
            # Check if idx is compatible with all already selected
            # selected is a boolean vector
            # We need to check if any selected node is incompatible with idx
            # incompat_sup[idx] is a vector of length N_sup
            # conflict = incompat_sup[idx] & selected
            # if not conflict.any():
            #    selected[idx] = True
            
            # Optimization: maintain a "blocked" set?
            # Or just compute. N_sup is 330. 330*330 operations per restart is small.
            if not np.any(incompat_sup[idx] & selected):
                selected[idx] = True
        
        size = np.sum(selected)
        if size > best_size:
            best_size = size
            best_iset = selected.copy()
    
    if best_iset is None:
        # Fallback to weight-2 or something
        return np.array([[]], dtype=np.int8)
        
    # Generate all weight-4 vectors for the selected supports
    vectors = []
    for i in range(N_sup):
        if best_iset[i]:
            sup = supports[i]
            for signs in product([1, -1], repeat=4):
                v = np.zeros(D, dtype=np.int8)
                for idx, s in zip(sup, signs):
                    v[idx] = s
                vectors.append(v)
                
    if not vectors:
         # Fallback
         return np.array([[]], dtype=np.int8)

    return np.array(vectors, dtype=np.int8)
