import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # Seed solution: N=2
    seed_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                seed_vectors.append(v)
    seed_array = np.array(seed_vectors, dtype=np.int64)
    best_k = len(seed_array)
    best_result = seed_array
    
    # Try N=4 search
    # Candidates: 4 non-zero entries, each +/- 1
    # Total candidates: 2^4 * C(11, 4) = 16 * 330 = 5280
    
    # Pre-compute candidates
    candidates = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates)
    num_candidates = len(candidates)
    
    # Greedy search function
    def greedy_search(order):
        selected = []
        selected_squares = [] # squared norms, all 4
        for idx in order:
            v = candidates[idx]
            # Check against all selected
            # dist^2 = ||v||^2 + ||u||^2 - 2 v.u = 8 - 2 v.u
            # Need dist^2 >= 4 => 8 - 2 v.u >= 4 => 2 v.u <= 4 => v.u <= 2
            if not selected:
                selected.append(v)
                continue
            
            # Vectorized dot product check
            # selected is list of vectors. Convert to array for speed?
            # For small k, loop is fine.
            valid = True
            for u in selected:
                dot = np.dot(v, u)
                if dot > 2:
                    valid = False
                    break
            if valid:
                selected.append(v)
        
        return selected

    # Run multiple random restarts
    np.random.seed(42)
    max_restarts = 50
    for i in range(max_restarts):
        if time.time() > deadline:
            break
            
        order = np.random.permutation(num_candidates)
        selected = greedy_search(order)
        
        if len(selected) > best_k:
            best_k = len(selected)
            best_result = np.array(selected, dtype=np.int64)
            
        # If we found a very good solution, we might stop early?
        # Or continue to find even better.
        
    return best_result
