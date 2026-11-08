import numpy as np
from itertools import combinations

D = 11

def _generate_weight_k_vectors(k):
    """Generate all integer vectors in R^D with exactly k non-zero entries, each +/- 1."""
    vectors = []
    for pos in combinations(range(D), k):
        for signs in range(2**k):
            v = np.zeros(D, dtype=np.int64)
            for i, p in enumerate(pos):
                v[p] = 1 if (signs >> i) & 1 else -1
            vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def _greedy_pack(vectors, max_dot=1):
    """
    Greedily select a subset of vectors such that pairwise dot product <= max_dot.
    """
    if len(vectors) == 0:
        return []
    
    # Compute all pairwise dot products
    # vectors is (N, D)
    # dots is (N, N)
    dots = vectors @ vectors.T
    
    # We want to select indices such that dots[i][j] <= max_dot for all i != j
    # And dots[i][i] is norm squared, which is > max_dot always.
    
    selected = []
    remaining = list(range(len(vectors)))
    
    # Heuristic: pick the vector with the smallest sum of dot products to others?
    # Or just random shuffle and greedy.
    # Let's try a few random shuffles.
    
    best_set = []
    rng = np.random.default_rng(42)
    
    for _ in range(5): # 5 trials
        perm = rng.permutation(len(vectors))
        current_set = []
        current_indices = []
        
        # Precompute compatibility? Too expensive for N=5000.
        # Just check against current set.
        
        for idx in perm:
            v = vectors[idx]
            compatible = True
            for j in current_indices:
                if dots[idx][j] > max_dot:
                    compatible = False
                    break
            if compatible:
                current_set.append(v)
                current_indices.append(idx)
                
        if len(current_set) > len(best_set):
            best_set = current_set
            
    return np.array(best_set, dtype=np.int64)

def construct():
    # 1. Weight 2 vectors
    v2 = _generate_weight_k_vectors(2)
    # All weight 2 vectors have dot product <= 1?
    # Max dot product for distinct weight 2 vectors is 1 (if they share 1 index with same sign, and 0 for the other? No.
    # u=e1+e2, v=e1+e3. dot=1.
    # u=e1+e2, v=e1-e2. dot=0.
    # u=e1+e2, v=-e1-e2. dot=-2.
    # Max is 1.
    # So all 220 are valid.
    best_vectors = v2
    best_k = len(v2)
    
    # 2. Weight 3 vectors
    v3 = _generate_weight_k_vectors(3)
    # Condition: dot <= 1
    # Greedy pack
    packed_v3 = _greedy_pack(v3, max_dot=1)
    if len(packed_v3) > best_k:
        best_vectors = packed_v3
        best_k = len(packed_v3)
        
    # 3. Weight 4 vectors
    v4 = _generate_weight_k_vectors(4)
    # Condition: dot <= 2
    # Note: Norm squared is 4.
    # dist^2 >= 4 => 8 - 2 dot >= 4 => dot <= 2.
    packed_v4 = _greedy_pack(v4, max_dot=2)
    if len(packed_v4) > best_k:
        best_vectors = packed_v4
        best_k = len(packed_v4)
        
    return best_vectors
