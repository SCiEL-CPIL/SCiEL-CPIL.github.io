import time
import numpy as np
from scipy.optimize import minimize

N = 50

def _compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.array(h)
    if np.all(h == 0):
        return 0.0
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    # Linear convolution (autoconvolution)
    conv = np.convolve(h, h, mode='full')
    l_inf = np.max(conv)
    if l_inf == 0:
        return 0.0
    return l2_sq / (l1 * l_inf)

def _find_sidon_set(n, max_iter=1000):
    """
    Find a large Sidon set in Z_n: a subset where all pairwise sums are distinct.
    This minimizes peak autoconvolution by ensuring each lag has at most one contributing pair.
    """
    import random
    random.seed(42)
    
    best_set = set()
    best_size = 0
    
    for attempt in range(max_iter):
        # Greedy construction
        current = set()
        sums = set()
        
        # Try adding elements in random order
        candidates = list(range(n))
        random.shuffle(candidates)
        
        for i in candidates:
            # Check if adding i creates any duplicate sums
            new_sums = set()
            valid = True
            for j in current:
                s = (i + j) % n
                if s in sums or s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            
            # Also check i+i
            s_ii = (2 * i) % n
            if s_ii in sums or s_ii in new_sums:
                valid = False
            
            if valid:
                current.add(i)
                sums.update(new_sums)
                sums.add(s_ii)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current.copy()
            
        # Early termination if we found a good set
        if best_size >= 8:  # Heuristic: 8 is quite good for n=50
            break
    
    return list(best_set)

def _sidon_to_heights(sidon_indices, n=N):
    """Convert a Sidon set to a height vector with small weights on non-Sidon positions"""
    h = np.zeros(n)
    for idx in sidon_indices:
        h[idx] = 1.0
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin within 120s limit
    
    best_h = None
    best_ratio = 0.0
    
    # Step 1: Find a good Sidon set
    sidon = _find_sidon_set(N, max_iter=200)
    
    # Step 2: Initialize from Sidon set and optimize
    h0 = _sidon_to_heights(sidon)
    h0 = h0 / np.sum(h0)  # Normalize
    
    # Also try spike initialization
    h_spike = np.zeros(N)
    h_spike[N // 2] = 1.0
    
    # Step 3: Use L-BFGS-B to optimize
    def negative_ratio(h):
        return -_compute_ratio(h)
    
    constraints = []
    bounds = [(0.0, None) for _ in range(N)]
    
    # Try multiple initializations
    inits = [h0, h_spike]
    
    # Add a few random sparse inits
    np.random.seed(123)
    for _ in range(3):
        h_rand = np.random.exponential(0.5, N)
        h_rand = h_rand / np.sum(h_rand)
        inits.append(h_rand)
    
    for init in inits:
        if time.time() > deadline - 15:
            break
        
        try:
            result = minimize(
                negative_ratio,
                init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            ratio = -result.fun
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = result.x.copy()
                
        except Exception:
            continue
    
    # Final check: ensure we have a valid result
    if best_h is None:
        best_h = h_spike
        best_ratio = _compute_ratio(best_h)
    
    # Convert to list as required
    return list(best_h)
