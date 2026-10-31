import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0
    D = 11

    # Generate all integer vectors with L2 norm <= 3 (norm^2 <= 9)
    # Norm^2 = 1: 22 vectors (11 axes * 2 signs)
    # Norm^2 = 2: 110 vectors (C(11,2) * 4 signs)
    # Norm^2 = 4: 5280 vectors (C(11,4) * 16 signs)
    # Norm^2 = 5: 9240 vectors (C(11,2) * C(9,3) * 8 signs) -- weight 5 with one 2 and four 1s? No, norm^2=5 means one 2 and one 1, or five 1s.
    # Let's stick to norm^2 <= 4 first to keep the pool manageable but large.
    # Pool size: 22 + 110 + 5280 = 5412.
    
    all_vectors = []
    
    # Norm^2 = 1
    for i in range(D):
        for s in [1, -1]:
            v = [0]*D
            v[i] = s
            all_vectors.append(v)
            
    # Norm^2 = 2
    for i in range(D):
        for j in range(i+1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0]*D
                    v[i] = si
                    v[j] = sj
                    all_vectors.append(v)
                    
    # Norm^2 = 4
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0]*D
            for pos, sign in zip(support, signs):
                v[pos] = sign
            all_vectors.append(v)
            
    # Also add Norm^2 = 5 vectors? (one 2, one 1)
    # This adds 11*10*4 = 440 vectors.
    # Let's include them.
    for i in range(D):
        for j in range(D):
            if i == j: continue
            for si in [1, -1]: # sign for the 2
                for sj in [1, -1]: # sign for the 1
                    v = [0]*D
                    v[i] = 2 * si
                    v[j] = sj
                    all_vectors.append(v)
                    
    # Also add Norm^2 = 5 vectors (five 1s)
    # C(11,5) * 32 = 462 * 32 = 14784. This is getting large.
    # Let's stick to norm^2 <= 4 for now. 5412 vectors.
    # Actually, let's add the (2,1) type ones. Total ~5852.
    
    all_vectors_np = np.array(all_vectors, dtype=np.int8)
    n_total = len(all_vectors_np)
    
    # Precompute squared norms
    norms_sq = np.sum(all_vectors_np ** 2, axis=1)
    
    # We want min pairwise distance >= max norm.
    # dist^2 = ||v||^2 + ||u||^2 - 2 v.u
    # Constraint: ||v||^2 + ||u||^2 - 2 v.u >= max_norm^2
    # If we assume all selected vectors have norm^2 = R (or close), then 2R - 2 v.u >= R => R >= 2 v.u => v.u <= R/2.
    # But norms vary.
    # Let M = max norm^2 in selected set.
    # Constraint: ||v||^2 + ||u||^2 - 2 v.u >= M.
    
    # Strategy:
    # 1. Sort vectors by norm descending? Or random?
    # 2. Greedy selection.
    
    rng = np.random.default_rng(42)
    best_selected = []
    best_score = 0 # k
    
    # We will try a few different greedy strategies.
    # Strategy 1: Random order, greedy check.
    # To speed up, we can group by support or just brute force with numpy.
    # n_total ~ 5852.
    # If we select k vectors, the check for a new vector is O(k).
    # Total O(k^2) checks? No, O(n_total * k).
    # If k ~ 100, n_total * k ~ 500k dot products. Very fast.
    # If k ~ 500, 500 * 5852 ~ 3M dot products. Still fast.
    
    # Let's try multiple random restarts.
    max_restarts = 50
    
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        # Random permutation of indices
        order = rng.permutation(n_total)
        
        selected_indices = []
        selected_vectors = []
        max_norm_sq = 0
        
        for idx in order:
            v = all_vectors_np[idx]
            v_norm_sq = norms_sq[idx]
            
            # New max norm if we add this
            new_max_norm_sq = max(max_norm_sq, v_norm_sq)
            
            compatible = True
            # Check against all selected vectors
            if selected_vectors:
                selected_arr = np.array(selected_vectors)
                # Compute dot products
                dots = selected_arr @ v
                # Compute constraint: ||u||^2 + v_norm_sq - 2*dots >= new_max_norm_sq
                # selected_norms_sq + v_norm_sq - 2*dots >= new_max_norm_sq
                # We need to store norms of selected vectors too.
                
                # Let's store selected norms in a list/array
                # For efficiency in the loop, let's keep selected_vectors as a list and convert to array for batch dot product.
                # This is O(k) memory and O(k) compute per candidate.
                
                # To optimize, we can maintain a running array of selected vectors.
                pass # Placeholder for logic
                
            # Actually, let's implement the loop properly.
            
            if not selected_vectors:
                selected_vectors.append(v)
                selected_indices.append(idx)
                max_norm_sq = v_norm_sq
                continue
            
            sel_arr = np.array(selected_vectors)
            sel_norms = np.sum(sel_arr ** 2, axis=1) # Recomputing is wasteful, but k is small.
            
            dots = sel_arr @ v
            lhs = sel_norms + v_norm_sq - 2 * dots
            if np.any(lhs < new_max_norm_sq):
                compatible = False
            
            if compatible:
                selected_vectors.append(v)
                selected_indices.append(idx)
                if v_norm_sq > max_norm_sq:
                    max_norm_sq = v_norm_sq
        
        k = len(selected_indices)
        if k > best_score:
            best_score = k
            best_selected = selected_indices
            # Optional: if best_score is high enough, break?
            # Kissing number in 11 is 512.
            # If we get close, maybe try to improve further?
            # For now, just keep going until time runs out.
            
    if not best_selected:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    return all_vectors_np[best_selected].astype(np.int64)
