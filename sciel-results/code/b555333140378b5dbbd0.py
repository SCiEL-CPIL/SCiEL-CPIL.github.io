import numpy as np
import time
import random
from itertools import combinations

D = 11

def construct():
    start_time = time.time()
    deadline = 130  # Safety margin before 140s kill
    
    # Generate candidate vectors
    # We consider vectors with small integer entries.
    # Norm squared must be consistent for pairwise distance check.
    # Condition: ||v_i - v_j|| >= ||v_max||
    # If all vectors have the same norm R, then ||v_i - v_j||^2 = 2R^2 - 2 v_i.v_j >= R^2
    # => 2 v_i.v_j <= R^2
    # => v_i.v_j <= R^2 / 2
    
    # Let's try R^2 = 4 (vectors with two 2s, or one 2 and two 1s? No, integer).
    # Vectors with norm squared 4:
    # Type A: (±2, 0, ..., 0) -> 2*11 = 22 vectors
    # Type B: (±1, ±1, 0, ..., 0) -> C(11,2)*4 = 55*4 = 220 vectors
    # Total 242 vectors.
    # Check condition for Type A vs Type A:
    # v1=(2,0..), v2=(2,0..) -> dot=4. R^2/2 = 2. 4<=2 False.
    # v1=(2,0..), v2=(-2,0..) -> dot=-4. -4<=2 True.
    # So we can't take all Type A. We can take at most one from each axis pair?
    # Actually, for Type A, v_i.v_j = 0 if different axes. 0 <= 2 True.
    # So we can take one vector from each axis, choosing sign.
    # But we also need to check against Type B.
    
    # Let's try R^2 = 5.
    # Type A: (±2, ±1, 0, ...) -> 11*10*4 = 440 vectors
    # Type B: (±1, ±1, ±1, ±1, ±1, 0...) no, 1^2+1^2+1^2+1^2+1^2 = 5.
    # C(11,5)*32 = 462*32 = 14784 vectors.
    # This is getting large.
    
    # Let's stick to R^2 = 3 first, which was in the current code.
    # Vectors with norm squared 3: (±1, ±1, ±1, 0, ...)
    # Number: C(11,3)*8 = 165*8 = 1320.
    # Condition: v_i.v_j <= 3/2 = 1.5. Since dot products are integers, v_i.v_j <= 1.
    
    # Let's generate these candidates.
    candidates = []
    for i, j, k in combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
    
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # Pre-compute Gram matrix G where G[i,j] = candidates[i] . candidates[j]
    # We need G[i,j] <= 1 for all i != j in the selected set.
    G = candidates @ candidates.T
    
    # Greedy algorithm with random restarts
    best_set = []
    best_size = 0
    
    # Heuristic: Sort candidates by some metric? Or just random permutation.
    # Random permutation is simple.
    
    num_trials = 0
    while time.time() < deadline - 5:
        order = np.random.permutation(n_cand)
        selected_indices = []
        
        for idx in order:
            if time.time() > deadline - 2:
                break
            
            if not selected_indices:
                selected_indices.append(idx)
            else:
                # Check if idx is compatible with all selected
                # G[idx, selected_indices] <= 1
                # Note: G[idx, idx] is 3, but we only check against selected_indices
                # which does not include idx yet.
                if np.all(G[idx, selected_indices] <= 1):
                    selected_indices.append(idx)
        
        if len(selected_indices) > best_size:
            best_size = len(selected_indices)
            best_set = selected_indices
            num_trials += 1
            
            # If we find a very large set, we might be close to optimal.
            # But let's keep searching for a bit.
        
        # To speed up subsequent trials, we could use a smarter order,
        # but random is robust.
        
    if best_size == 0:
        # Fallback to a known valid small set
        # (1,1,1,0..), (-1,-1,1,0..), etc.
        # Just return one vector to be safe
        return np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    return candidates[best_set]
