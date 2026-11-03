import itertools
import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # 1. Generate seed vectors (2-sparse, +/-1)
    seed_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                seed_vectors.append(v)
    
    # 2. Generate candidate 3-sparse vectors
    # Types: (1, -1, -1) permutations and (-1, -1, -1)
    candidates = []
    for i, j, k in itertools.combinations(range(D), 3):
        # Type 1: One +1, two -1s
        # +1 at i
        v1 = [0]*D; v1[i]=1; v1[j]=-1; v1[k]=-1
        candidates.append(v1)
        # +1 at j
        v2 = [0]*D; v2[i]=-1; v2[j]=1; v2[k]=-1
        candidates.append(v2)
        # +1 at k
        v3 = [0]*D; v3[i]=-1; v3[j]=-1; v3[k]=1
        candidates.append(v3)
        # Type 2: Three -1s
        v4 = [0]*D; v4[i]=-1; v4[j]=-1; v4[k]=-1
        candidates.append(v4)
        
    # 3. Build conflict graph among candidates
    # Conflict if dot product >= 2
    n_cand = len(candidates)
    # Adjacency list or matrix
    # 660x660 is small
    conflict = np.zeros((n_cand, n_cand), dtype=bool)
    
    # Vectorized dot product check
    # Candidates are 0/+-1 vectors
    # We can compute all pairwise dot products
    cand_arr = np.array(candidates)
    dots = cand_arr @ cand_arr.T
    
    # Conflict if dot >= 2
    conflict = dots >= 2
    np.fill_diagonal(conflict, False)
    
    # 4. Find Maximum Independent Set (MIS)
    # Greedy heuristic first
    # Then maybe simple local search
    
    # Greedy: pick node with min degree, remove it and neighbors
    # Repeat
    remaining = set(range(n_cand))
    mis = []
    
    # Pre-compute degrees
    degrees = conflict.sum(axis=1)
    
    # Simple greedy
    while remaining:
        # Find node with min degree in remaining
        min_deg = float('inf')
        best_node = -1
        for node in remaining:
            deg = sum(1 for nb in remaining if conflict[node, nb])
            if deg < min_deg:
                min_deg = deg
                best_node = node
        
        if best_node == -1: break
        mis.append(best_node)
        # Remove best_node and its neighbors
        to_remove = set()
        to_remove.add(best_node)
        for nb in remaining:
            if conflict[best_node, nb]:
                to_remove.add(nb)
        remaining -= to_remove
        
        if time.time() > deadline:
            break
            
    # Convert MIS indices to vectors
    selected_candidates = [candidates[i] for i in mis]
    
    # Combine
    all_vectors = seed_vectors + selected_candidates
    
    return np.array(all_vectors, dtype=np.int64)
