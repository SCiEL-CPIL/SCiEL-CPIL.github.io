import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin: kill at 150s, return by 140s

    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    candidates = []

    def update_best(h, sc):
        nonlocal best_h, best_score
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
        candidates.append((sc, h.copy()))

    # Phase 1: Sparse Random Search
    # Hypothesis: Sparse supports with low collision (Sidon-like) yield higher ratios.
    # We generate random sparse vectors with support size k in [3, 20].
    
    np.random.seed(42) # For reproducibility in debugging, though random is fine
    
    num_sparse_tries = 2000
    for _ in range(num_sparse_tries):
        if time.time() > deadline - 30.0:
            break
        
        # Choose support size
        k = np.random.randint(3, 16)
        # Choose random indices
        indices = np.random.choice(N, k, replace=False)
        
        # Create sparse vector with random positive weights
        weights = np.random.rand(k)
        h = np.zeros(N)
        h[indices] = weights
        
        sc = compute_ratio(h)
        update_best(h, sc)
        
    # Phase 1b: Structured Sparse "Sidon" Attempts
    # Try specific patterns that might minimize collisions
    # 1. Arithmetic progression (bad for collisions, but let's check)
    # 2. Quadratic residues / Squares
    for shift in range(N):
        if time.time() > deadline - 30.0:
            break
        h = np.zeros(N)
        for i in range(int(np.sqrt(N)) + 2):
            idx = (i*i + shift) % N
            h[idx] = 1.0
        if np.sum(h) > 0:
            sc = compute_ratio(h)
            update_best(h, sc)
            
    # 3. Pairs of primes / random primes
    # Just random sparse is likely sufficient, but let's add a few manual sparse patterns
    # e.g. indices 0, 1, 3, 7, 15 (powers of 2 - 1)
    patterns = [
        [0, 1, 3, 7, 15, 31],
        [0, 2, 5, 11, 23],
        [0, 1, 4, 9, 16, 25, 36, 49],
        [0, 10, 20, 30, 40],
        [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
    ]
    for pat in patterns:
        if time.time() > deadline - 30.0:
            break
        h = np.zeros(N)
        for idx in pat:
            if idx < N:
                h[idx] = 1.0
        sc = compute_ratio(h)
        update_best(h, sc)

    # Sort candidates by score and keep top K for local refinement
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_k = candidates[:20]
    
    # Phase 2: Local Optimization from Top Candidates
    # We optimize the weights of the sparse supports.
    # Note: The support is fixed during optimization to preserve the "sparse" property,
    # or we can allow small perturbations. Nelder-Mead on full vector might collapse to dense.
    # Let's optimize only the non-zero elements of the top candidates.
    
    def objective_neg_sparse(h_sparse, indices):
        # h_sparse is the vector of weights for the given indices
        h = np.zeros(N)
        h[indices] = h_sparse
        h = np.maximum(h, 0.0)
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        h = h / s
        h2 = np.sum(h**2)
        conv = fftconvolve(h, h, mode='full')
        cinf = np.max(conv)
        if cinf < 1e-15:
            return 1e10
        return -h2 / cinf

    for i, (sc, h_start) in enumerate(top_k):
        if time.time() > deadline - 10.0:
            break
        
        # Identify non-zero indices
        indices = np.where(h_start > 1e-10)[0]
        if len(indices) == 0:
            continue
            
        h_sparse_start = h_start[indices]
        
        # Optimize weights for this fixed support
        res = minimize(objective_neg_sparse, h_sparse_start, args=(indices,), 
                       method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        h_opt_sparse = np.maximum(res.x, 0.0)
        h_opt = np.zeros(N)
        h_opt[indices] = h_opt_sparse
        
        s = np.sum(h_opt)
        if s > 1e-15:
            h_opt = h_opt / s
            sc_opt = compute_ratio(h_opt)
            update_best(h_opt, sc_opt)
            
        # Also try full vector optimization for the very best few, 
        # but constrain to non-negative. This might find dense solutions if they are better,
        # but given the "Sparse Sidon" insight, sparse should be better.
        # Let's do one full optimization on the absolute best sparse candidate found so far.
        if i == 0:
            def objective_neg_full(h_flat):
                h = np.maximum(h_flat, 0.0)
                s = np.sum(h)
                if s < 1e-15:
                    return 1e10
                h = h / s
                h2 = np.sum(h**2)
                conv = fftconvolve(h, h, mode='full')
                cinf = np.max(conv)
                if cinf < 1e-15:
                    return 1e10
                return -h2 / cinf
                
            h_full_start = h_start.copy()
            res_full = minimize(objective_neg_full, h_full_start, 
                               method='Nelder-Mead', 
                               options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            h_full_opt = np.maximum(res_full.x, 0.0)
            s_full = np.sum(h_full_opt)
            if s_full > 1e-15:
                h_full_opt = h_full_opt / s_full
                sc_full = compute_ratio(h_full_opt)
                update_best(h_full_opt, sc_full)

    return list(best_h)
