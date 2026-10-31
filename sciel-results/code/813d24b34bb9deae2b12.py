import numpy as np
import itertools
import time

D = 11

def generate_candidates():
    vectors = []
    for combo in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            v[combo[0]] = signs[0]
            v[combo[1]] = signs[1]
            vectors.append(v)
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(combo, signs):
                v[idx] = s
            vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    candidates = generate_candidates()
    n = len(candidates)
    
    if n == 0:
        return np.empty((0, D), dtype=np.int64)
    
    norm_sq = np.sum(candidates**2, axis=1)
    dots = candidates @ candidates.T
    min_norm_sq = np.minimum(norm_sq[:, None], norm_sq[None, :])
    conflict = dots > (min_norm_sq / 2.0)
    np.fill_diagonal(conflict, False)
    
    # Precompute neighbor lists for faster updates
    # conflict is a boolean matrix
    # neighbors[i] is an array of indices j such that conflict[i,j] is True
    neighbors = [np.where(conflict[i])[0] for i in range(n)]
    
    degrees = np.array([len(neigh) for neigh in neighbors])
    
    best_result = np.empty((0, D), dtype=np.int64)
    best_size = 0
    
    for seed in range(2000):
        if time.time() > deadline:
            break
        
        rng = np.random.default_rng(seed)
        noise = rng.random(n) * 10
        order = np.argsort(degrees + noise, kind='stable')
        
        selected = []
        selected_bool = np.zeros(n, dtype=bool)
        conflict_count = np.zeros(n, dtype=np.int32)
        
        for idx in order:
            if time.time() > deadline:
                break
            
            if selected_bool[idx]:
                continue
            if conflict_count[idx] > 0:
                continue
            
            # Select idx
            selected.append(int(idx))
            selected_bool[idx] = True
            
            # Update conflict counts for neighbors
            neigh = neighbors[idx]
            conflict_count[neigh] += 1
        
        cur_size = len(selected)
        if cur_size > best_size:
            best_size = cur_size
            best_result = candidates[np.array(selected, dtype=np.int64)]
            
    return best_result
