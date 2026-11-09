import numpy as np
import itertools
import time

D = 11

def generate_q2_vectors():
    """Generate all 220 weight-2 vectors (norm squared 2)."""
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0]*D
                v[i], v[j] = si, sj
                vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def generate_q4_core_vectors():
    """Generate 5280 weight-4 vectors with 4 non-zeros (norm squared 4)."""
    vectors = []
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0]*D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def generate_q4_sat_vectors():
    """Generate 22 vectors of form 2*e_i (norm squared 4)."""
    vectors = []
    for i in range(D):
        for s in (2, -2):
            v = [0]*D
            v[i] = s
            vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def find_max_clique(candidates, deadline):
    """
    Find a large clique in the compatibility graph of candidates.
    Compatibility: dot(u, v) <= Q/2. Here Q=4, so dot <= 2.
    """
    N = len(candidates)
    if N == 0:
        return []
    
    # Compute dot products
    M = candidates @ candidates.T
    
    # Threshold for Q=4 is 2.0
    threshold = 2.0
    
    # Build adjacency matrix (boolean)
    # A[i,j] = True if i != j and M[i,j] <= threshold
    A = (M <= threshold)
    np.fill_diagonal(A, False)
    
    # Greedy Max Clique
    degrees = np.sum(A, axis=1)
    order = np.argsort(-degrees)
    
    remaining = set(range(N))
    clique = []
    
    for v in order:
        if time.time() > deadline:
            break
        if v not in remaining:
            continue
        
        clique.append(v)
        
        # Remove v and its non-neighbors (conflicts)
        non_neigh = np.where(~A[v])[0]
        remaining.difference_update(non_neigh)
        
    return clique

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 20s margin
    
    # 1. Baseline: Q=2 (220 vectors)
    q2_vectors = generate_q2_vectors()
    best_vectors = q2_vectors
    best_score = len(q2_vectors)
    
    # 2. Try Q=4 Core + Satellites
    if time.time() < deadline:
        core = generate_q4_core_vectors()
        sat = generate_q4_sat_vectors()
        
        clique_indices = find_max_clique(core, deadline)
        
        if len(clique_indices) > 0:
            # The core clique is compatible with all satellites
            # We can add all satellites
            core_clique = core[clique_indices]
            combined = np.vstack([core_clique, sat])
            
            # Verify a few random pairs to be safe?
            # The logic holds: dot(core, sat) in {-2,0,2} <= 2.
            # dot(sat, sat) is 0 for distinct.
            
            if len(combined) > best_score:
                best_score = len(combined)
                best_vectors = combined

    if len(best_vectors) == 0:
        # Fallback to seed (should not happen)
        return generate_q2_vectors()
        
    return best_vectors
