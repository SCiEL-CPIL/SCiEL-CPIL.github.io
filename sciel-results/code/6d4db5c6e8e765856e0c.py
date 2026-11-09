import numpy as np
import itertools
import time

D = 11
DEADLINE = 115.0

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time

    # Part 1: All 220 vectors of Type 1
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
    
    # Part 2: Type 2 candidates (Even parity)
    # 165 supports, 128 sign patterns each
    candidates = []
    for support in itertools.combinations(range(D), 8):
        # Generate 128 even parity sign patterns
        # Iterate over first 7 signs, determine 8th
        for signs_7 in itertools.product([-1, 1], repeat=7):
            # Product of first 7
            prod_7 = 1
            for s in signs_7:
                prod_7 *= s
            # 8th sign must be prod_7 to make total product 1
            sign_8 = prod_7
            signs = list(signs_7) + [sign_8]
            
            v = [0] * D
            for k, idx in enumerate(support):
                v[idx] = signs[k]
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int8)
    n_cand = len(candidates)
    
    # Compute conflict matrix
    # C[i, j] = 1 if candidates[i] . candidates[j] > 4
    candidates_int = candidates.astype(np.int16)
    dots = candidates_int @ candidates_int.T
    conflict = dots > 4
    
    # Multi-start Greedy MIS
    rng = np.random.default_rng(42)
    best_selected = []
    best_size = 0
    
    # We can run multiple iterations
    # Estimate time per iteration:
    # N=21120. Loop N times.
    # Each step: boolean mask update.
    # Should be < 0.1s per iteration.
    # Let's try 200 iterations.
    
    max_iters = 500
    for _ in range(max_iters):
        if elapsed() > DEADLINE:
            break
            
        order = rng.permutation(n_cand)
        selected = []
        available = np.ones(n_cand, dtype=bool)
        
        for idx in order:
            if not available[idx]:
                continue
                
            # Select idx
            selected.append(idx)
            
            # Mark idx and its neighbors as unavailable
            available[idx] = False
            available[conflict[idx]] = False
            
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
            
    # Build final list
    final_vectors = vectors.copy()
    if best_selected:
        selected_vecs = candidates[best_selected]
        final_vectors.extend(selected_vecs.tolist())
        
    return np.array(final_vectors, dtype=np.int64)
