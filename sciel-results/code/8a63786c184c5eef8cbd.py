import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    degrees = [len(n) for n in neighbors]
    
    best_set = None
    best_size = 0
    rng = np.random.default_rng(42)
    
    # Try multiple random greedy runs
    for run in range(50):
        if time.time() > deadline - 20:
            break
            
        alive = np.ones(N4, dtype=bool)
        selected = []
        degs = np.array(degrees, dtype=np.int32)
        
        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = np.min(degs[alive_indices])
            cands = alive_indices[degs[alive_indices] == min_deg]
            if len(cands) > 1:
                chosen = cands[rng.integers(len(cands))]
            else:
                chosen = cands[0]
            selected.append(chosen)
            to_remove = np.unique(np.concatenate(([chosen], neighbors[chosen])))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degs[rem] -= np.sum(conflicts[rem][:, to_remove], axis=1)
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
            
    result_v4 = vectors_v4[best_set]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
