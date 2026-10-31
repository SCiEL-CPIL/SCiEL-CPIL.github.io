import numpy as np
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # Generate all norm squared 3 vectors: +/- e_i +/- e_j +/- e_k
    # i < j < k
    vectors = []
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            vectors.append(v)
    
    vectors = np.array(vectors)
    # 1320 vectors
    
    # Greedy selection
    # We want a subset S such that u . v <= 1 for all u, v in S.
    # Note: u . u = 3.
    
    # Randomize order
    indices = np.arange(len(vectors))
    
    best_set = []
    best_size = 0
    
    # Try a few random shuffles
    for attempt in range(10):
        if time.time() > deadline:
            break
            
        np.random.shuffle(indices)
        current_set = []
        current_vectors = []
        
        for idx in indices:
            v = vectors[idx]
            # Check against current set
            if not current_vectors:
                current_set.append(idx)
                current_vectors.append(v)
                continue
                
            # Dot products
            # v is (1,1,1) permuted.
            # u is (1,1,1) permuted.
            # Dot product is sum of products of coordinates.
            # Max dot product is 3 (if same).
            # We need dot <= 1.
            
            dots = np.sum(current_vectors * v, axis=1)
            if np.max(dots) <= 1:
                current_set.append(idx)
                current_vectors.append(v)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    if best_size > 220:
        return vectors[best_set]
    else:
        # Fallback to seed
        import itertools
        seed_vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    seed_vectors.append(v)
        return np.array(seed_vectors, dtype=np.int64)
