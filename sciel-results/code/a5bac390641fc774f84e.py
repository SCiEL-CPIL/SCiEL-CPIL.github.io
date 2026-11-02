import itertools
import numpy as np
import time

D = 11

def generate_candidates():
    # N2: 2 non-zero entries
    n2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                n2.append(v)
                
    # N3: 3 non-zero entries
    n3 = []
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    n3.append(v)
                    
    # N4: 4 non-zero entries
    n4 = []
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        n4.append(v)
                        
    return np.array(n2), np.array(n3), np.array(n4)

def greedy_pack(candidates, min_dist_sq, seed=0, time_limit=10):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(candidates))
    rng.shuffle(idx)
    
    selected = []
    start_time = time.time()
    
    for i in idx:
        if time.time() - start_time > time_limit:
            break
        v = candidates[i]
        if not selected:
            selected.append(v)
            continue
        
        # Check distance to all selected
        # dist_sq = ||v - s||^2
        # Use vectorized calculation
        diffs = np.array(selected) - v
        dists_sq = np.sum(diffs * diffs, axis=1)
        if np.all(dists_sq >= min_dist_sq):
            selected.append(v)
            
    return np.array(selected) if selected else np.empty((0, D), dtype=np.int64)

def construct():
    n2, n3, n4 = generate_candidates()
    
    best_set = np.empty((0, D), dtype=np.int64)
    
    # Case 1: N2 + N3, min_dist_sq = 3
    pool1 = np.vstack([n2, n3])
    for seed in range(20):
        res = greedy_pack(pool1, 3, seed=seed, time_limit=5)
        if len(res) > len(best_set):
            best_set = res
            
    # Case 3: N2 + N3 + N4, min_dist_sq = 4
    pool3 = np.vstack([n2, n3, n4])
    for seed in range(50):
        res = greedy_pack(pool3, 4, seed=seed, time_limit=5)
        if len(res) > len(best_set):
            best_set = res
            
    return best_set
