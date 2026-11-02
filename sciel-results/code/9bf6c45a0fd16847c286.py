import itertools
import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit

    # Helper to check if a vector can be added to a set
    # Constraint: min pairwise distance >= max norm
    # If all vectors have norm 2 (squared norm 4), then dist2 >= 4
    # dist2 = |u-v|^2 = |u|^2 + |v|^2 - 2u.v = 4 + 4 - 2u.v = 8 - 2u.v
    # 8 - 2u.v >= 4  =>  4 >= 2u.v  =>  u.v <= 2
    
    def can_add(vec, selected):
        if not selected:
            return True
        # Compute dot products with all selected vectors
        # selected is a list of 1D arrays
        dots = np.array([np.dot(v, vec) for v in selected])
        return np.all(dots <= 2)

    # Strategy:
    # 1. Generate a large pool of candidates with norm 2 (squared norm 4).
    #    - Weight 2 vectors: (±1, ±1, 0, ...) -> C(11,2)*4 = 220 vectors
    #    - Weight 4 vectors: (±1, ±1, ±1, ±1, 0, ...) -> C(11,4)*16 = 5280 vectors
    #    - Weight 11 vectors: (±1, ..., ±1) with even number of -1s? 
    #      Norm^2 = 11. This is not norm 2. So we stick to norm 2 vectors for the main set.
    #      Wait, the problem says "minimum pairwise distance is at least the maximum vector norm".
    #      If we mix norms, it's complicated. 
    #      Let's stick to norm 2 vectors for simplicity and high density.
    #      Actually, D11 lattice vectors are typically norm 2.
    #      Let's generate all weight-2 and weight-4 vectors.
    
    candidates = []
    
    # Weight 2 vectors
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
                    
    # Weight 4 vectors
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates)
    
    best_set = []
    
    # Multiple random restarts to escape local minima of greedy selection
    rng = np.random.default_rng(42)
    
    # Try different orderings
    # 1. Deterministic order (original)
    # 2. Random shuffles
    
    for restart in range(100):
        if time.time() > deadline - 10.0:
            break
            
        if restart == 0:
            order = np.arange(n_candidates)
        else:
            order = rng.permutation(n_candidates)
            
        selected = []
        
        for i in order:
            if time.time() > deadline - 5.0:
                break
                
            c = candidates[i]
            
            # Optimization: Vectorized check if possible?
            # For now, loop is okay if we break early or keep it fast.
            # np.dot with a list of arrays is slow.
            # Let's store selected in a 2D array for faster dot product?
            # But selected grows.
            
            if not selected:
                selected.append(c)
                continue
            
            # Check compatibility
            # Convert selected to array for fast dot product
            # This is O(k) per candidate, where k is current size.
            # Total O(N * K^2) roughly.
            
            # To speed up, we can pre-compute dot products with the first few selected?
            # Or just use the list comprehension which is implemented in C for np.dot?
            # Actually, np.dot(selected_arr, c) is much faster if selected is a 2D array.
            # But appending to a 2D array is expensive.
            
            # Let's try a hybrid: keep a list, but periodically convert?
            # Or just accept the list overhead for small K.
            # K can be up to ~500.
            # 5000 candidates * 500 checks * 11 dims = 27.5M ops. Should be fast enough in Python/NumPy if vectorized.
            
            # Let's use a 2D array for selected to leverage BLAS for dot products.
            # We'll maintain a list for easy appending, but if we convert every time, it's O(K^2) memory copy.
            
            # Alternative: Use a list of vectors, and check one by one.
            # If we find a violation, break early.
            
            compatible = True
            for v in selected:
                if np.dot(v, c) > 2:
                    compatible = False
                    break
                    
            if compatible:
                selected.append(c)
                
        if len(selected) > len(best_set):
            best_set = selected
            # Early termination if we reach a known good size?
            # The kissing number in 11 is not known exactly, but D11 gives 560?
            # Actually, the D_n lattice has 2n(n-1) vectors of norm 2.
            # For n=11, that is 2*11*10 = 220 vectors.
            # But we can add more?
            # The A_n lattice?
            # The current program score 1.2682 suggests it found more than the seed (220).
            # 1.2682 * 220 = 279.
            # Inspiration 1 (score 1.8909) found 1.8909 * 220 = 416.
            # So we should aim for > 416.
            
            if len(best_set) >= 500:
                # Check if we can improve further or if we're hitting a wall
                pass

    # If we didn't find anything (shouldn't happen), fallback
    if not best_set:
        # Return the first few weight-2 vectors
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j] = si, sj
                    vectors.append(v)
        best_set = vectors

    return np.array(best_set, dtype=np.int64)
