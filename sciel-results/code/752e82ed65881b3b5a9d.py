import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11
    
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)
    sup_sets = [set(s) for s in supports]
    
    # Precompute intersection sizes
    inter = np.zeros((N_sup, N_sup), dtype=np.int8)
    for i in range(N_sup):
        for j in range(i + 1, N_sup):
            si = len(sup_sets[i] & sup_sets[j])
            inter[i, j] = si
            inter[j, i] = si
    
    rng = np.random.default_rng(42)
    best_clique = None
    best_size = 0
    
    # Phase 1: Greedy with random restarts
    for r in range(200):
        if time.time() > deadline - 40.0:
            break
        order = rng.permutation(N_sup)
        selected = set()
        for idx in order:
            ok = True
            for s in selected:
                if inter[idx, s] > 2:
                    ok = False
                    break
            if ok:
                selected.add(idx)
        if len(selected) > best_size:
            best_size = len(selected)
            best_clique = selected.copy()
    
    if best_clique is None:
        return np.array([[]], dtype=np.int8)
    
    # Phase 2: Local search - try to add non-conflicting supports
    improved = True
    while improved and time.time() < deadline - 20.0:
        improved = False
        current = best_clique.copy()
        for candidate in range(N_sup):
            if candidate in current:
                continue
            # Check if compatible with all in current
            ok = True
            for s in current:
                if inter[candidate, s] > 2:
                    ok = False
                    break
            if ok:
                current.add(candidate)
                improved = True
        if improved:
            best_clique = current
            best_size = len(current)
    
    # Generate vectors
    vectors = []
    for i in best_clique:
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)
    
    if not vectors:
        return np.array([[]], dtype=np.int8)
    
    return np.array(vectors, dtype=np.int8)
