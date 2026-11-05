import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    # Start with the seed solution
    h = np.ones(N)
    best_val = 2.0
    best_h = h.copy()
    
    # We work in normalized space (sum=1) to simplify
    # But the evaluator uses raw h.
    # Since value is scale invariant, we can optimize normalized h and return it.
    # Or just return h.
    
    def get_value(h_vec):
        s = np.sum(h_vec)
        if s == 0:
            return np.inf
        conv = np.convolve(h_vec, h_vec)
        max_c = np.max(conv)
        return abs(2 * N * max_c / (s ** 2))
    
    # Normalize initial h
    h_norm = h / np.sum(h)
    best_val_norm = get_value(h_norm)
    best_h_norm = h_norm.copy()
    
    alpha = 0.05
    # Try a few different starting points?
    # Just run the loop.
    
    while time.time() - start_time < time_limit:
        # Generate perturbation
        # Use a mix of small and large steps
        scale = np.random.choice([0.01, 0.05, 0.1])
        d = np.random.normal(0, 1, N)
        h_try = best_h_norm + scale * d
        
        s_try = np.sum(h_try)
        if s_try == 0:
            continue
            
        h_try = h_try / s_try
        
        val_try = get_value(h_try)
        
        if val_try < best_val_norm:
            best_val_norm = val_try
            best_h_norm = h_try.copy()
            
    # Return the best h found
    # The evaluator accepts any finite h with sum != 0.
    # Returning normalized h is fine.
    return best_h_norm
