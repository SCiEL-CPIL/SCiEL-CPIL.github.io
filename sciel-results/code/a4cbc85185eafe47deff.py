import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_h = np.ones(N)
    best_val = 2.0  # seed value
    
    # Try a few random strategies
    # 1. Random normal, scaled to have sum N
    # 2. Random sparse
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Generate random h
        # Ensure sum is not zero
        h = rng.standard_normal(N)
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        
        # Normalize to sum = N to keep scale comparable?
        # V is scale invariant. 
        # But numerical stability?
        # Let's just compute V directly.
        
        c = np.convolve(h, h)
        max_c = np.max(c)
        val = abs(2 * N * max_c / (s * s))
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
        # Try a structured perturbation of ones
        # h = 1 + eps * noise
        eps = 0.1 * (1.0 - (time.time() - start_time) / 140.0)
        h2 = 1.0 + eps * rng.standard_normal(N)
        s2 = np.sum(h2)
        if abs(s2) < 1e-6:
            continue
        c2 = np.convolve(h2, h2)
        max_c2 = np.max(c2)
        val2 = abs(2 * N * max_c2 / (s2 * s2))
        
        if val2 < best_val:
            best_val = val2
            best_h = h2.copy()
            
    return best_h
