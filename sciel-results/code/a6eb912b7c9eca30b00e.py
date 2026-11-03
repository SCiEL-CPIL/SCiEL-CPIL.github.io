import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    D = 11
    
    # Type A: Vectors with one non-zero entry +/- 2
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
    # Generate all supports (4-subsets)
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)
    
    # Map support to index
    support_to_idx = {s: i for i, s in enumerate(supports)}
    
    # Build conflict graph for supports
    # Conflict if intersection size >= 3
    conflict_supports = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter_len = len(set(supports[i]) & set(supports[j]))
            if inter_len >= 3:
                conflict_supports[i, j] = True
                conflict_supports[j, i] = True
                
    # Find maximum independent set of supports
    # Using greedy with random restarts
    best_support_set = []
    best_size = 0
    
    # Precompute degrees
    degrees = np.sum(conflict_supports, axis=1)
    
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline - 30: # Leave time for vector generation
            break
            
        available = set(range(N_S))
        current_set = []
        
        # Shuffle available for randomization
        avail_list = list(available)
        np.random.shuffle(avail_list)
        
        # Sort by degree (ascending) to pick low-degree nodes first?
        # Or just pick randomly from available?
        # Let's try picking min degree from available
        while available:
            # Find node with min degree in available
            # N_S is 330, so scanning is fast
            min_deg = float('inf')
            v_idx = -1
            for v in avail_list:
                if v in available:
                    d = degrees[v]
                    if d < min_deg:
                        min_deg = d
                        v_idx = v
            
            if v_idx == -1: break
            
            current_set.append(v_idx)
            available.remove(v_idx)
            
            # Remove neighbors
            neighbors = np.where(conflict_supports[v_idx])[0]
            available.difference_update(neighbors)
            
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_support_set = current_set
            
    # Generate Type D vectors for the selected supports
    selected_vectors_D = []
    for s_idx in best_support_set:
        support = supports[s_idx]
        # Generate all 16 sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
            
    selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)
    
    # Try to add more Type D vectors from non-selected supports
    # This is a greedy fill
    current_set_vectors = list(selected_vectors_D)
    current_set_vectors_set = set(map(tuple, current_set_vectors))
    
    # We need to check compatibility with ALL current vectors
    # To speed this up, we can just check against the selected vectors.
    # Since N_S is small, we can iterate through all 5280 vectors and check.
    
    # Actually, let's just generate ALL Type D vectors and check if they can be added.
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
            
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)
    
    # Precompute dot products? 5280x5280 is 28M. Might be slow.
    # Instead, for each candidate, check against current set.
    # Current set size is ~ 15 * 16 = 240.
    # 5280 * 240 = 1.2M dot products. Fast.
    
    added_indices = set()
    for i in range(N_all_D):
        if time.time() > deadline - 10:
            break
            
        v = all_vectors_D[i]
        # Check if v is already in set
        if tuple(v) in current_set_vectors_set:
            continue
            
        can_add = True
        # Check against current_set_vectors
        for u in current_set_vectors:
            dot = np.dot(v, u)
            if dot >= 3:
                can_add = False
                break
                
        if can_add:
            current_set_vectors.append(v)
            current_set_vectors_set.add(tuple(v))
            
    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    
    # Combine Type A and Type D
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    
    return final_vectors
