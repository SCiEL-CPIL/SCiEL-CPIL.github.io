import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Generate candidates
    candidates = []
    
    # Norm squared 2
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0]*D
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
                    
    # Norm squared 3
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = [0]*D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
                            
    N = len(candidates)
    candidates = np.array(candidates, dtype=np.int64)
    norms_sq = np.sum(candidates**2, axis=1)
    
    # 2. Build conflict graph
    dots = candidates @ candidates.T
    max_norm_sq = np.maximum(norms_sq[:, None], norms_sq[None, :])
    threshold = max_norm_sq / 2.0
    conflict = (dots > threshold).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    best_set = []
    best_size = 0
    
    # 3. Randomized Greedy Multiple Times
    # We have ~1540 candidates.
    # Each greedy pass is O(N^2) roughly.
    # 1540^2 = 2.4M. Fast.
    # We can run ~100-500 passes.
    
    for _ in range(500):
        if time.time() > deadline - 5:
            break
            
        order = np.random.permutation(N)
        current_set = []
        current_mask = np.zeros(N, dtype=bool)
        
        for idx in order:
            # Check conflict
            if not np.any(conflict[idx, current_mask]):
                current_set.append(idx)
                current_mask[idx] = True
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    # Final local search on the best set found
    if best_set:
        current_mask = np.zeros(N, dtype=bool)
        for idx in best_set:
            current_mask[idx] = True
            
        # Try to add more
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(N):
                if current_mask[i]: continue
                if not np.any(conflict[i, current_mask]):
                    current_mask[i] = True
                    best_set.append(i)
                    improved = True
                    
    final_vectors = candidates[best_set]
    return final_vectors
