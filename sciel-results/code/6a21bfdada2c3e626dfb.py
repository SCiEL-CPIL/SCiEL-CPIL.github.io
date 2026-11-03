import itertools
import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Generate weight-4 vectors
    # v has 4 non-zero entries, each 1 or -1
    candidates = []
    indices = list(range(D))
    
    # To speed up, we can pre-compute dot products or use bitmasks
    # But 5280 vectors is small enough for brute force dot products
    
    for comb in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            candidates.append(v)
            
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # Compute all pairwise dot products
    # This is 5280 x 5280 matrix. ~28M entries. 
    # Memory: 28M * 8 bytes = 224 MB. Acceptable?
    # Time: 28M ops. Fast.
    
    # However, we only need to check if dot > 2 (conflict for R=4)
    # Condition: dist^2 >= 4 => 8 - 2 dot >= 4 => 2 dot <= 4 => dot <= 2.
    # Conflict if dot >= 3.
    
    # Greedy selection
    selected = []
    selected_indices = []
    
    # Randomize order to avoid bias
    np.random.seed(42)
    order = np.random.permutation(n_cand)
    
    for idx in order:
        if time.time() > deadline:
            break
            
        v = candidates[idx]
        # Check against all selected
        if not selected:
            selected.append(v)
            selected_indices.append(idx)
            continue
            
        # Vectorized check
        # dots = np.dot(selected, v)
        # if np.all(dots <= 2):
        #     selected.append(v)
        #     selected_indices.append(idx)
        
        # Optimization: keep a list of selected vectors as an array
        # This might be slow if we append to list and convert.
        # Better to maintain a list and check one by one?
        # Or maintain a matrix?
        
        # Let's use a simple loop for now, it's O(N^2) total.
        # 5000 * 500 = 2.5M ops. Fast.
        
        is_ok = True
        for sv in selected:
            if np.dot(sv, v) > 2:
                is_ok = False
                break
        if is_ok:
            selected.append(v)
            selected_indices.append(idx)
            
    # Local Search Improvement
    # Try to replace selected vectors with non-selected ones to allow more?
    # Or just add more if possible?
    # The greedy order matters.
    
    # Let's try a few random restarts if time permits
    best_k = len(selected)
    best_vectors = np.array(selected)
    
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Try 10 random restarts
        for restart in range(10):
            if time.time() > deadline - 2:
                break
            np.random.seed(42 + restart)
            order = np.random.permutation(n_cand)
            selected = []
            for idx in order:
                if time.time() > deadline - 2:
                    break
                v = candidates[idx]
                if not selected:
                    selected.append(v)
                    continue
                is_ok = True
                for sv in selected:
                    if np.dot(sv, v) > 2:
                        is_ok = False
                        break
                if is_ok:
                    selected.append(v)
            
            if len(selected) > best_k:
                best_k = len(selected)
                best_vectors = np.array(selected)
                
    return best_vectors
