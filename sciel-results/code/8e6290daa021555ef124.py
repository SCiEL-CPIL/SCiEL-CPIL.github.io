import numpy as np
import random

def _eval_norm(coeffs):
    """
    Computes the sup norm of the polynomial P(z) = sum c_k z^k for |z|=1.
    Uses dense sampling and golden section refinement for accuracy.
    """
    n = len(coeffs)
    N = 2048
    z = np.exp(2j * np.pi * np.arange(N) / N)
    powers = np.vander(z, n, increasing=True).T
    P = coeffs @ powers
    
    mags = np.abs(P)
    idx = np.argsort(mags)[-10:]
    max_val = 0.0
    
    # Precompute coeffs for speed in inner loop?
    # coeffs is a numpy array.
    
    for i in idx:
        theta = 2 * np.pi * i / N
        a = theta - np.pi / N
        b = theta + np.pi / N
        
        # Use a faster evaluation for golden section
        # Horner's rule is O(n).
        # We can precompute powers of z_val? No, z_val changes.
        
        def f(theta_val):
            z_val = np.exp(1j * theta_val)
            val = 0.0
            z_pow = 1.0
            # Unroll or use dot product with precomputed powers?
            # Powers of z_val: 1, z, z^2...
            # Since n=24, a simple loop is fast.
            for c in coeffs:
                val += c * z_pow
                z_pow *= z_val
            return abs(val)
        
        gr = (np.sqrt(5) - 1) / 2
        c_val = b - gr * (b - a)
        d_val = a + gr * (b - a)
        fc = f(c_val)
        fd = f(d_val)
        
        for _ in range(50):
            if fc < fd:
                b = d_val
                d_val = c_val
                fd = fc
                c_val = b - gr * (b - a)
                fc = f(c_val)
            else:
                a = c_val
                c_val = d_val
                fc = fd
                d_val = a + gr * (b - a)
                fd = f(d_val)
        
        peak_val = max(fc, fd)
        if peak_val > max_val:
            max_val = peak_val
            
    return max_val

def _local_search(coeffs, max_iters=1000):
    """
    Improved local search: Single flips, then pair flips.
    """
    current_max = _eval_norm(coeffs)
    n = len(coeffs)
    
    # Phase 1: Single bit flips
    improved = True
    while improved and max_iters > 0:
        improved = False
        best_flip_idx = -1
        best_new_max = current_max
        best_new_coeffs = coeffs.copy()
        
        for k in range(n):
            new_coeffs = coeffs.copy()
            new_coeffs[k] = -new_coeffs[k]
            new_max = _eval_norm(new_coeffs)
            
            if new_max < best_new_max - 1e-12:
                best_new_max = new_max
                best_flip_idx = k
                best_new_coeffs = new_coeffs
            max_iters -= 1
            if max_iters <= 0:
                break
        
        if best_flip_idx != -1:
            coeffs = best_new_coeffs
            current_max = best_new_max
            improved = True
    
    if max_iters <= 0:
        return coeffs, current_max

    # Phase 2: Pair bit flips
    improved = True
    while improved and max_iters > 0:
        improved = False
        best_flip_idx1 = -1
        best_flip_idx2 = -1
        best_new_max = current_max
        best_new_coeffs = coeffs.copy()
        
        # Sample a subset of pairs to save time if n is large, but n=24 is small
        # 24*23/2 = 276 pairs. 276 evals per iteration.
        # If we do 10 iterations, 2760 evals.
        # Each eval ~0.1ms? 0.2s. OK.
        
        for k1 in range(n):
            for k2 in range(k1 + 1, n):
                new_coeffs = coeffs.copy()
                new_coeffs[k1] = -new_coeffs[k1]
                new_coeffs[k2] = -new_coeffs[k2]
                
                new_max = _eval_norm(new_coeffs)
                
                if new_max < best_new_max - 1e-12:
                    best_new_max = new_max
                    best_flip_idx1 = k1
                    best_flip_idx2 = k2
                    best_new_coeffs = new_coeffs
                max_iters -= 1
                if max_iters <= 0:
                    break
            if max_iters <= 0:
                break
        
        if best_flip_idx1 != -1:
            coeffs = best_new_coeffs
            current_max = best_new_max
            improved = True
            
    return coeffs, current_max

def construct():
    # Generate initial candidates
    candidates = []
    
    # 1. Truncated GRS
    p, q = [1], [1]
    while len(p) < 48: # Make it longer to allow shifts
        p, q = p + q, p + [-c for c in q]
    
    # Take first 24
    g24 = np.array(p[:24], dtype=float)
    candidates.append(g24)
    
    # 2. Reversed GRS
    candidates.append(g24[::-1].copy())
    
    # 3. Shifted GRS (slide window)
    for shift in range(1, 24):
        candidates.append(np.array(p[shift:shift+24], dtype=float))
        
    # 4. Random sequences
    random.seed(42)
    for _ in range(10):
        candidates.append(np.array([1 if random.random() < 0.5 else -1 for _ in range(24)], dtype=float))

    best_coeffs = None
    best_score = float('inf')
    
    # Time limit management
    import time
    start_time = time.time()
    time_limit = 110.0 # Leave 10s for overhead
    
    for cand in candidates:
        if time.time() - start_time > time_limit:
            break
            
        # Run local search
        # Allocate remaining time proportionally or fixed budget per candidate
        # With ~30 candidates, we can give each ~3s?
        # Or just run until no improvement.
        # Local search converges fast for single flips.
        
        res_coeffs, res_score = _local_search(cand.copy(), max_iters=2000)
        
        if res_score < best_score:
            best_score = res_score
            best_coeffs = res_coeffs
            
    if best_coeffs is None:
        # Fallback
        best_coeffs = g24
        
    return [int(c) for c in best_coeffs]
