import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1

    # Build conflict matrix for weight-4 vectors
    # Conflict if dot product > 2
    # Since max norm is 4 (squared), distance squared = 8 - 2*dot.
    # dist^2 >= 4 => 8 - 2*dot >= 4 => dot <= 2.
    # So conflict if dot > 2.
    
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists for faster greedy update
    # Using flatnonzero for speed
    neighbors = [np.flatnonzero(conflicts[i]) for i in range(N4)]
    
    best_indices = []
    best_size = 0
    rng = np.random.default_rng(42)
    
    # Run multiple random restarts of the greedy independent set algorithm
    # The greedy algorithm:
    # 1. Randomly permute vertices.
    # 2. Iterate through permutation.
    # 3. If vertex is active (not excluded by previous selections), select it.
    # 4. Deactivate it and its neighbors.
    
    n_restarts = 200 # Start with 200, can adjust if time permits
    
    for _ in range(n_restarts):
        if time.time() > deadline - 5:
            break
            
        active = np.ones(N4, dtype=bool)
        selected = []
        order = rng.permutation(N4)
        
        for v in order:
            if active[v]:
                selected.append(v)
                active[v] = False
                # Deactivate neighbors
                nbrs = neighbors[v]
                if len(nbrs) > 0:
                    active[nbrs] = False
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_indices = selected
            # Optional: break if we hit a theoretical upper bound? 
            # Or just keep trying to find larger.
            
    # Combine with weight-1 vectors
    # Weight-1 vectors are compatible with all weight-4 vectors (max dot 2)
    # and with each other (dot 0 or 4? No, distinct weight-1 vectors have dot 0. Identical not included).
    # Wait, distinct weight-1 vectors: u=(2,0..), v=(-2,0..). dot=-4?
    # u=(2,0..), v=(0,2..). dot=0.
    # u=(2,0..), v=(2,0..) -> identical, not in set.
    # So all distinct weight-1 vectors are compatible.
    # And weight-1 vs weight-4: max dot is 2. Compatible.
    
    result_v4 = vectors_v4[best_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

def _fallback_with_mis(vectors_v4, vectors_v1, deadline):
    """Fallback if the math assumption is wrong."""
    # This function is kept for compatibility but the main logic is in construct
    return vectors_v1
