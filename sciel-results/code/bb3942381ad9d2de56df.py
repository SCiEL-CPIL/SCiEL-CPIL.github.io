import numpy as np
import itertools
import time

def generate_candidates():
    d = 11
    candidates = []
    
    # Norm squared 2: two +-1
    for i in range(d):
        for j in range(i+1, d):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
                    
    # Norm squared 4
    # Type A: one +-2
    for i in range(d):
        for s in [-2, 2]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Type B: four +-1
    for indices in itertools.combinations(range(d), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for idx, s in zip(indices, signs):
                v[idx] = s
            candidates.append(v)
            
    return np.array(candidates)

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590 # 10s margin
    
    candidates = generate_candidates()
    n_cand = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Greedy selection
    best_set = []
    best_size = 0
    
    # We can try multiple random permutations to find a better greedy solution
    # Or sort by "conflict degree"
    
    # Let's compute a heuristic score: number of conflicts.
    # Conflict if not compatible.
    # Compatible(u, v) <=> dist2 > max(norm_u, norm_v)
    
    # This might be slow to compute for all pairs (5500^2 ~ 30M pairs).
    # 30M operations in Python loop is too slow.
    # We need to vectorize or be smarter.
    
    # However, we don't need the full matrix. We can compute degrees on the fly or use a faster method.
    # Or we can just use random shuffles. 5500 candidates.
    # Greedy pass: O(N^2) per pass? No, checking compatibility with selected set is O(K) per candidate, where K is size of selected set.
    # Total O(N*K). K is ~200. N=5500. 1.1M ops per pass.
    # We can do many passes.
    
    # Let's try 100 random passes.
    best_points = None
    
    for run in range(200):
        if time.time() > deadline:
            break
            
        # Random shuffle
        indices = np.random.permutation(n_cand)
        
        selected = []
        selected_indices = []
        
        for idx in indices:
            v = candidates[idx]
            norm_v = norms_sq[idx]
            
            is_compatible = True
            # Check against all selected
            for s_idx in selected_indices:
                u = candidates[s_idx]
                norm_u = norms_sq[s_idx]
                dot = np.dot(u, v)
                dist_sq = norm_u + norm_v - 2 * dot
                if dist_sq <= max(norm_u, norm_v):
                    is_compatible = False
                    break
            
            if is_compatible:
                selected.append(v)
                selected_indices.append(idx)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_points = selected
            
    if best_points is None:
        # Fallback to seed
        return np.array([np.ones(11), -1 * np.ones(11)]).astype(np.int64)
        
    return np.array(best_points, dtype=np.int64)
