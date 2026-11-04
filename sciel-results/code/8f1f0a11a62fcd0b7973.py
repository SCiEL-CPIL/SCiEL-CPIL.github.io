import numpy as np
import itertools
import random
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit to be safe
    
    # Generate all vectors of norm sqrt(3)
    # These are vectors with exactly three non-zero entries, each +/-1
    vectors = []
    indices = range(D)
    for support in itertools.combinations(indices, 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            vectors.append(v)
            
    vectors = np.array(vectors, dtype=np.int64)
    n = len(vectors)
    
    # Precompute supports (tuples of indices with non-zero values)
    supports = []
    for v in vectors:
        supp = tuple(i for i in range(D) if v[i] != 0)
        supports.append(supp)
        
    # Conflict check: two supports conflict if their intersection size is 1
    # We want to find a large subset of indices such that for any pair, intersection size != 1
    
    best_set = []
    best_score = 0
    
    # Greedy with random restarts
    # We can speed up by checking conflicts against the current set
    
    for restart in range(50): # Limit restarts
        if time.time() > deadline:
            break
            
        random.shuffle(vectors)
        random.shuffle(supports)
        
        current_set = []
        current_supports = []
        
        for i in range(n):
            if time.time() > deadline:
                break
                
            supp = supports[i]
            conflict = False
            for existing_supp in current_supports:
                # Check intersection size
                inter = set(supp).intersection(existing_supp)
                if len(inter) == 1:
                    conflict = True
                    break
            if not conflict:
                current_set.append(vectors[i])
                current_supports.append(supp)
                
        if len(current_set) > best_score:
            best_score = len(current_set)
            best_set = current_set[:]
            
    if not best_set:
        # Fallback to the 220 vectors of norm sqrt(2)
        vecs = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vecs.append(v)
        return np.array(vecs, dtype=np.int64)
        
    return np.array(best_set, dtype=np.int64)
