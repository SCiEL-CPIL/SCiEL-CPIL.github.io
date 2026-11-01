import numpy as np
import itertools
import time

D = 11

def generate_candidates(max_norm_sq=3):
    candidates = []
    # Norm squared 1: one non-zero +-1
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Norm squared 2: two non-zero +-1
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Norm squared 3: three non-zero +-1
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
                    
    return candidates

def is_compatible(u, v, max_norm_sq_u, max_norm_sq_v):
    # u, v are integer vectors
    # d^2 >= max(||u||^2, ||v||^2)
    diff = u - v
    d_sq = np.dot(diff, diff)
    max_n_sq = max(max_norm_sq_u, max_norm_sq_v)
    return d_sq >= max_n_sq

def greedy_construct(candidates, order=None, max_iter=None):
    if order is None:
        order = np.arange(len(candidates))
    else:
        order = order[:max_iter] if max_iter else order
        
    selected = []
    selected_norms_sq = []
    
    for idx in order:
        c = candidates[idx]
        c_norm_sq = np.dot(c, c)
        compatible = True
        for i, v in enumerate(selected):
            if not is_compatible(c, v, c_norm_sq, selected_norms_sq[i]):
                compatible = False
                break
        if compatible:
            selected.append(c)
            selected_norms_sq.append(c_norm_sq)
            
    if not selected:
        return np.array([[]], dtype=np.int64)
        
    return np.array(selected, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    candidates = generate_candidates(max_norm_sq=3)
    n_cand = len(candidates)
    
    best_set = None
    best_size = 0
    
    # Try a few specific orders and random orders
    # 1. All norm 2 (the seed)
    # We know this yields 220.
    
    # 2. Random shuffles
    np.random.seed(42)
    
    for i in range(50): # 50 restarts
        if time.time() > deadline:
            break
            
        order = np.random.permutation(n_cand)
        # We can also try sorting by norm
        # But random is good for diversity
        
        res = greedy_construct(candidates, order)
        size = res.shape[0]
        
        if size > best_size:
            best_size = size
            best_set = res
            
        # Optimization: if we hit a very high number, we can stop early?
        # Or just keep going.
        
    # Also try a deterministic "small norm first" or "large norm first"
    # Small norm first might allow more large norms?
    # Large norm first might block small norms?
    
    # Let's try one run sorted by norm squared ascending
    norms = [np.dot(c, c) for c in candidates]
    order_asc = np.argsort(norms)
    res_asc = greedy_construct(candidates, order_asc)
    if res_asc.shape[0] > best_size:
        best_size = res_asc.shape[0]
        best_set = res_asc
        
    # Let's try one run sorted by norm squared descending
    order_desc = np.argsort(norms)[::-1]
    res_desc = greedy_construct(candidates, order_desc)
    if res_desc.shape[0] > best_size:
        best_size = res_desc.shape[0]
        best_set = res_desc
        
    if best_set is None:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        best_set = np.array(vectors, dtype=np.int64)
        
    return best_set
