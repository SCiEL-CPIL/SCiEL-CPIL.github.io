import numpy as np
import itertools
import random
import time

def kissing_number11() -> np.ndarray:
    d = 11
    vectors = []
    for support in itertools.combinations(range(d), 3):
        for signs in itertools.product([-1, 1], repeat=3):
            v = np.zeros(d, dtype=np.int64)
            for idx, val in zip(support, signs):
                v[idx] = val
            vectors.append(v)
    
    N = len(vectors)
    
    vec_to_idx = {}
    for i, v in enumerate(vectors):
        vec_to_idx[tuple(v)] = i
        
    neighbors = [[] for _ in range(N)]
    
    for i, v in enumerate(vectors):
        support = [j for j in range(d) if v[j] != 0]
        for pair in itertools.combinations(support, 2):
            i1, i2 = pair
            s1, s2 = v[i1], v[i2]
            for l in range(d):
                if l in support: continue
                for s3 in [-1, 1]:
                    u = np.zeros(d, dtype=np.int64)
                    u[i1] = s1
                    u[i2] = s2
                    u[l] = s3
                    j = vec_to_idx.get(tuple(u), -1)
                    if j != -1 and j != i:
                        neighbors[i].append(j)
                        
    best_set = []
    deadline = time.time() + 580
    
    # Use a faster random greedy
    for restart in range(200):
        if time.time() > deadline: break
        
        active = [True] * N
        selected = []
        
        # To speed up random selection, we can just pick a random index and check if active
        # But that's slow if many are inactive.
        # Better: maintain a list of active indices.
        active_indices = list(range(N))
        random.shuffle(active_indices)
        
        while active_indices:
            # Pick first element (random due to shuffle)
            i = active_indices.pop(0)
            if not active[i]:
                continue
            active[i] = False
            selected.append(i)
            
            # Remove neighbors
            for j in neighbors[i]:
                if active[j]:
                    active[j] = False
                    # We don't remove from active_indices immediately, 
                    # we just skip them in the loop.
                    # This makes the list long with inactive elements.
                    # But pop(0) is O(N).
                    
        # This pop(0) strategy is O(N^2) per restart.
        # 1320^2 * 200 = 348 * 10^6. Should be ok.
        
        if len(selected) > len(best_set):
            best_set = selected
            
    if not best_set:
        # Fallback
        points = []
        for i in range(d):
            for j in range(i+1, d):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        p = np.zeros(d, dtype=np.int64)
                        p[i] = s1
                        p[j] = s2
                        points.append(p)
        return np.array(points, dtype=np.int64)
        
    result = np.array([vectors[i] for i in best_set], dtype=np.int64)
    return result
