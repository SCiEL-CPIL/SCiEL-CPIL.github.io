import numpy as np
import time
import random

D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate candidate vectors
    # S1: weight 1
    # S2: weight 2
    # S3: weight 3
    
    candidates = []
    
    # S1
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # S2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # S3
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
                    
    candidates = np.array(candidates)
    norms_sq = np.sum(candidates**2, axis=1)
    
    best_vectors = []
    best_score = 0
    best_R_sq = 0
    
    n_candidates = len(candidates)
    
    # Randomized Greedy Search
    for run in range(2000):
        if time.time() > deadline:
            break
            
        # Shuffle indices
        idx = np.arange(n_candidates)
        np.random.shuffle(idx)
        
        V = []
        R_sq = 0
        
        for i in idx:
            v = candidates[i]
            v_norm_sq = norms_sq[i]
            
            new_R_sq = max(R_sq, v_norm_sq)
            limit = new_R_sq // 2
            
            if not V:
                V.append(v)
                R_sq = new_R_sq
            else:
                # Check compatibility
                # V is a list of arrays. Convert to array for vectorized dot product?
                # For efficiency, keep V as list of arrays, but check dot products.
                # If V is large, vectorize.
                # Let's convert V to array if it gets big?
                # Or just loop. K is small (~200).
                compatible = True
                for u in V:
                    dot = np.dot(v, u)
                    if dot > limit:
                        compatible = False
                        break
                if compatible:
                    V.append(v)
                    R_sq = new_R_sq
        
        score = len(V)
        if score > best_score:
            best_score = score
            best_vectors = V
            best_R_sq = R_sq
            
    if not best_vectors:
        # Fallback: return seed solution
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return np.array(best_vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
