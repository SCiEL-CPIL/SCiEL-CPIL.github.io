import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Generate vectors
    vectors = []
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            vectors.append(v)
            
    for indices in itertools.combinations(range(D), 4):
        i, j, k, l = indices
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    for s4 in [1, -1]:
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        v[k] = s3
                        v[l] = s4
                        vectors.append(v)
                        
    vectors = np.array(vectors, dtype=np.int64)
    n = len(vectors)
    
    # Precompute dot products
    # This might be memory heavy but 224MB is okay
    # dots = vectors @ vectors.T
    # To save memory/time, we can compute in blocks or just do it.
    # 5302x5302 is fine.
    
    dots = vectors @ vectors.T
    # Conflict if dot >= 3
    # Note: diagonal is 4, which is >= 3, but we don't pick same vertex twice.
    conflict = dots >= 3
    
    # Greedy with restarts
    best_set = []
    best_size = 0
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        available = np.ones(n, dtype=bool)
        selected = []
        
        while np.sum(available) > 0:
            if time.time() > deadline:
                break
            
            # Pick a random available vertex
            # To improve, we could pick the one with fewest conflicts among available
            # But random is faster.
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0:
                break
                
            # Heuristic: pick vertex with minimum degree in the induced subgraph?
            # Or just random.
            # Let's try random first.
            v_idx = avail_indices[np.random.randint(len(avail_indices))]
            
            selected.append(v_idx)
            
            # Mark v_idx and its neighbors as unavailable
            available[v_idx] = False
            available[conflict[v_idx]] = False
            
        size = len(selected)
        if size > best_size:
            best_size = size
            best_set = selected[:]
            
        if best_size >= 558:
            break
            
    if best_set:
        return vectors[best_set]
    else:
        # Fallback
        return np.array([[0]*D], dtype=np.int64)
