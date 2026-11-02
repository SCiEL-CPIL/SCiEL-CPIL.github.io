import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # 1. Generate Candidates: Weight-4 vectors (four +/-1s)
    # Symmetry breaking: First non-zero coordinate must be +1.
    # This reduces the search space by half compared to all weight-4 vectors.
    candidates = []
    
    for idxs in itertools.combinations(range(D), 4):
        # idxs is a tuple of 4 sorted indices
        # The first index in idxs is the "first non-zero"
        # We fix its sign to +1
        # The other 3 indices can be +1 or -1
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            v[idxs[0]] = 1
            for i, idx in enumerate(idxs[1:]):
                v[idx] = signs[i]
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # 2. Compute Conflict Matrix
    # Condition: ||vi - vj||^2 >= ||vk||^2
    # For weight-4 vectors, ||v||^2 = 4.
    # ||vi - vj||^2 = 4 + 4 - 2*dot(vi, vj) = 8 - 2*dot(vi, vj)
    # We need 8 - 2*dot(vi, vj) >= 4  =>  4 >= 2*dot(vi, vj) => dot(vi, vj) <= 2
    # Conflict if dot(vi, vj) > 2
    
    M = candidates @ candidates.T
    conflict = (M > 2).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    # 3. Greedy Min-Degree Selection
    # We want to pick a set of mutually compatible vectors.
    # Greedy strategy: At each step, pick the vector with the fewest conflicts
    # among the *remaining* candidates (or simply the one with min degree in the
    # current conflict graph induced by unselected nodes, but that's expensive).
    # A simpler robust heuristic: 
    # Initialize with empty set.
    # While possible:
    #   For each candidate v not in S:
    #     Check if v is compatible with all in S.
    #     If yes, calculate a "score" (e.g., number of conflicts with other unselected candidates).
    #     Pick the compatible v with the lowest score (min-degree in remaining graph).
    #     Add to S.
    
    selected = []
    in_S = np.zeros(N, dtype=bool)
    
    # Pre-compute initial degrees for tie-breaking or ordering if needed
    # But we need dynamic degrees.
    
    # To speed up, we can maintain a list of active candidates.
    # However, N=2640 is small.
    
    while True:
        if time.time() > deadline - 10:
            break
            
        # Find all candidates compatible with current S
        if len(selected) == 0:
            compatible_indices = np.where(~in_S)[0]
        else:
            # conflict[:, in_S] is a matrix of shape (N, len(selected))
            # We need all rows to be 0
            compat_mask = np.all(conflict[:, in_S] == 0, axis=1)
            compatible_indices = np.where(compat_mask & ~in_S)[0]
            
        if len(compatible_indices) == 0:
            break
            
        # For each compatible candidate, count conflicts with other compatible candidates?
        # Or just conflicts with all unselected candidates?
        # "Min-degree" usually means degree in the remaining graph.
        # Let's compute the degree of each compatible candidate in the subgraph of unselected nodes.
        
        # This is O(N * K) where K is size of compatible set.
        # N=2640, K can be up to 2640. 2640^2 ~ 7M ops. Fast enough for a few iterations.
        # But we do this loop potentially many times.
        
        # Optimization: We can just pick the one with the lowest degree in the *original* graph
        # among the compatible ones? Or random?
        # Let's try: Pick the compatible candidate with the minimum number of conflicts
        # with other *compatible* candidates. This tends to isolate the choice and
        # preserves options for others.
        
        # Extract sub-conflict matrix for compatible_indices
        if len(compatible_indices) > 1:
            sub_conflicts = conflict[np.ix_(compatible_indices, compatible_indices)]
            degrees = sub_conflicts.sum(axis=1)
            min_deg_idx = np.argmin(degrees)
            best_idx = compatible_indices[min_deg_idx]
        else:
            best_idx = compatible_indices[0]
            
        # Select best_idx
        selected.append(best_idx)
        in_S[best_idx] = True
        
    result = candidates[selected].astype(np.int64)
    
    # Final Verification
    if len(result) == 0:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    n = len(result)
    max_norm_sq = int(np.max(np.sum(result**2, axis=1)))
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            diff = result[i] - result[j]
            if np.dot(diff, diff) < max_norm_sq:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to safe small set (standard basis scaled by 2)
        vecs = []
        for i in range(D):
            v = [0]*D
            v[i] = 2
            vecs.append(v)
        return np.array(vecs, dtype=np.int64)
        
    return result
