import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # 1. Generate candidates
    candidates = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(tuple(v)) # tuple for hashing
    candidates = list(set(candidates)) # ensure unique
    N = len(candidates)
    
    # Map vector to index
    vec_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # 2. Precompute conflicts
    # Index by (3 indices, 3 signs)
    # Key: (idx1, idx2, idx3, s1, s2, s3) where idx1<idx2<idx3
    from collections import defaultdict
    groups = defaultdict(list)
    
    for i, v in enumerate(candidates):
        # v is a tuple of 11 ints
        # Get support
        support = [j for j in range(D) if v[j] != 0]
        # support has 4 elements
        # Generate 4 subsets of size 3
        for k in range(4):
            # remove support[k]
            sub_idx = support[:k] + support[k+1:]
            sub_signs = []
            for j in sub_idx:
                sub_signs.append(v[j])
            key = (tuple(sub_idx), tuple(sub_signs))
            groups[key].append(i)
            
    conflicts = [[] for _ in range(N)]
    for i, v in enumerate(candidates):
        support = [j for j in range(D) if v[j] != 0]
        for k in range(4):
            sub_idx = support[:k] + support[k+1:]
            sub_signs = []
            for j in sub_idx:
                sub_signs.append(v[j])
            key = (tuple(sub_idx), tuple(sub_signs))
            for j in groups[key]:
                if j != i:
                    conflicts[i].append(j)
    
    # Remove duplicates in conflicts (just in case)
    for i in range(N):
        conflicts[i] = list(set(conflicts[i]))
        
    best_count = 0
    best_set = []
    
    # 3. Greedy with restarts
    indices = list(range(N))
    
    while time.time() < deadline:
        random.shuffle(indices)
        blocked = np.zeros(N, dtype=bool)
        selected = []
        
        for idx in indices:
            if not blocked[idx]:
                selected.append(idx)
                # Block conflicts
                for c in conflicts[idx]:
                    blocked[c] = True
                    
        if len(selected) > best_count:
            best_count = len(selected)
            best_set = selected[:]
            
    if not best_set:
        return np.empty((0, D), dtype=np.int64)
        
    result = np.array([candidates[i] for i in best_set], dtype=np.int64)
    return result
