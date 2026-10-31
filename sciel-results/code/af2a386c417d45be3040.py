import itertools
import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Seed solution
    vectors_seed = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors_seed.append(v)
    seed_array = np.array(vectors_seed, dtype=np.int64)
    best_result = seed_array
    best_score = len(seed_array)
    
    # Try norm-sq-3 solution
    vectors_3 = []
    for comb in itertools.combinations(range(D), 3):
        i, j, k = comb
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    v[k] = s3
                    vectors_3.append(v)
                    
    V3 = np.array(vectors_3, dtype=np.int64)
    n3 = len(V3)
    
    if time.time() > deadline:
        return best_result
        
    Dot3 = V3 @ V3.T
    rows, cols = np.where(Dot3 > 1)
    
    conflicts = [set() for _ in range(n3)]
    for r, c in zip(rows, cols):
        if r != c:
            conflicts[r].add(c)
            conflicts[c].add(r)
            
    degrees = [len(c) for c in conflicts]
    order = np.argsort(degrees)
    
    selected = []
    removed = np.zeros(n3, dtype=bool)
    
    for node in order:
        if removed[node]:
            continue
        selected.append(node)
        removed[node] = True
        for neighbor in conflicts[node]:
            removed[neighbor] = True
            
    if time.time() > deadline:
        return best_result
        
    result_3 = np.array([vectors_3[i] for i in selected], dtype=np.int64)
    
    if len(result_3) > best_score:
        best_result = result_3
        best_score = len(result_3)
        
    return best_result
# EVOLVE-BLOCK-END
