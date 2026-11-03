import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    D = 11
    
    # Type A: Vectors with one non-zero entry +/- 2
    # Norm squared = 4
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
        
    vectors_A = np.array(vectors_A, dtype=np.int64)
    
    # Type D: Vectors with four non-zero entries +/- 1
    # Norm squared = 4
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports) # 330
    
    # Map support to index
    support_to_idx = {s: i for i, s in enumerate(supports)}
    
    # Build conflict graph for supports
    # Conflict if intersection size == 3
    # If intersection size is 4, they are the same support (not an edge in simple graph, handled by uniqueness)
    # If intersection size < 3, no conflict (compatible)
    conflict_supports = np.zeros((N_S, N_S), dtype=bool)
    
    # Precompute sets for faster intersection
    support_sets = [set(s) for s in supports]
    
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter_len = len(support_sets[i] & support_sets[j])
            if inter_len == 3:
                conflict_supports[i, j] = True
                conflict_supports[j, i] = True
                
    # Find Maximum Independent Set (MIS) of this graph
    best_set = []
    best_size = 0
    
    # Precompute degrees for initial greedy
    degrees = np.sum(conflict_supports, axis=1)
    
    # Greedy MIS with random restarts and local search
    num_restarts = 5000
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Randomize order slightly or use degree-based
        # Strategy: Sort by degree, but add random noise to break ties or perturb order
        # Or just random permutation
        order = np.random.permutation(N_S)
        
        selected_mask = np.zeros(N_S, dtype=bool)
        selected = []
        
        # Greedy Phase
        for idx in order:
            if selected_mask[idx]:
                continue
            # Check if idx conflicts with any selected
            # conflict_supports[idx] is a boolean array
            if not np.any(conflict_supports[idx] & selected_mask):
                selected.append(idx)
                selected_mask[idx] = True
                
        # Local Search Phase: 1-for-2 swaps
        improved = True
        while improved:
            improved = False
            # Try to add any node
            for idx in range(N_S):
                if selected_mask[idx]:
                    continue
                if not np.any(conflict_supports[idx] & selected_mask):
                    selected.append(idx)
                    selected_mask[idx] = True
                    improved = True
            
            # Try 1-for-2 swaps
            # For each u in selected, check if we can remove u and add two others
            # This is O(N^3) worst case, but N=330, so 330^3 ~ 35M. Might be slow in Python loop.
            # Optimize: Only try swaps if size is small or use numpy.
            # Given time limit, we might skip complex swaps if greedy is good enough.
            # However, 1-for-2 is powerful.
            # Let's try a simpler local search: check all unselected nodes against all selected nodes?
            # The "add any node" loop above is O(N^2).
            # The swap loop is expensive.
            # Let's stick to "add any node" and maybe "remove one, add two" if time permits.
            
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = selected.copy()
                
    # Generate Type D vectors for the selected supports
    selected_vectors_D = []
    for s_idx in best_set:
        support = supports[s_idx]
        # Generate all 16 sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
            
    selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)
    
    # Combine Type A and Type D
    final_vectors = np.vstack([vectors_A, selected_vectors_D])
    
    return final_vectors
