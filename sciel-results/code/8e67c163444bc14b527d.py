"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # 1. Generate Candidate Vectors
    # We use vectors with squared norm <= 4 to keep the number manageable and density high.
    candidates = []
    
    # Norm 1
    for i in range(D):
        v = np.zeros(D, dtype=np.int64)
        v[i] = 1
        candidates.append(v)
        v[i] = -1
        candidates.append(v)
        
    # Norm 2
    for i in range(D):
        for j in range(i+1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
                    
    # Norm 3
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        for sk in [1, -1]:
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
                            
    # Norm 4
    # Type 1: Weight 4, all entries +/-1
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for l in range(k+1, D):
                    for si in [1, -1]:
                        for sj in [1, -1]:
                            for sk in [1, -1]:
                                for sl in [1, -1]:
                                    v = np.zeros(D, dtype=np.int64)
                                    v[i] = si
                                    v[j] = sj
                                    v[k] = sk
                                    v[l] = sl
                                    candidates.append(v)
                                    
    # Type 2: Weight 1, entry +/-2
    for i in range(D):
        v = np.zeros(D, dtype=np.int64)
        v[i] = 2
        candidates.append(v)
        v[i] = -2
        candidates.append(v)
        
    C = np.array(candidates, dtype=np.int64)
    N = len(C)
    
    # 2. Build Incompatibility Graph
    norms_sq = np.sum(C**2, axis=1)
    dots = C @ C.T
    
    dist_sqs = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    max_norms = np.maximum(norms_sq[:, None], norms_sq[None, :])
    incompat = dist_sqs < max_norms
    
    np.fill_diagonal(incompat, False)
    
    # Convert to adjacency list
    neighbors = [set(np.where(incompat[i])[0]) for i in range(N)]
    
    # 3. Greedy Initialization
    degrees = np.array([len(neighbors[i]) for i in range(N)])
    order = np.argsort(degrees)
    
    current_set = set()
    for i in order:
        if i not in current_set and not (neighbors[i] & current_set):
            current_set.add(i)
            
    best_set = set(current_set)
    best_size = len(best_set)
    
    # 4. Local Search
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        improved = False
        candidates_to_try = rng.permutation(N)
        
        for u in candidates_to_try:
            if u in current_set:
                continue
                
            conflicts = neighbors[u] & current_set
            
            if len(conflicts) == 0:
                current_set.add(u)
                improved = True
            elif len(conflicts) == 1:
                v = list(conflicts)[0]
                # Try swap: remove v, add u
                # Only do this if it potentially helps (e.g., v has high degree)
                # For now, just do it to explore
                current_set.remove(v)
                current_set.add(u)
                improved = True # Mark as changed to continue search
                
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(best_set)
            
        if not improved:
            break
            
    if not best_set:
        return C[0].reshape(1, -1)
        
    result_indices = list(best_set)
    return C[result_indices]

# EVOLVE-BLOCK-END
