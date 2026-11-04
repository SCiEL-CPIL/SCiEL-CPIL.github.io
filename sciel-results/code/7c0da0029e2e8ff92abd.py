import numpy as np
import time

N = 50

def _compute_ratio(h):
    """
    Computes the ratio: ||h||_2^2 / (||h||_infty * ||h*h||_infty)
    Assuming h is normalized such that max(h) = 1.
    """
    if np.max(h) == 0:
        return 0.0
    h_norm = h / np.max(h)
    l2_sq = np.sum(h_norm ** 2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_max = np.max(conv)
    if conv_max == 0:
        return 0.0
    return l2_sq / conv_max

def _local_search_binary(h, max_iter=5000):
    """
    Improves a binary sequence by flipping bits to maximize the ratio.
    """
    best_h = h.copy()
    best_score = _compute_ratio(best_h)
    n = len(best_h)
    
    # Pre-compute convolution update logic might be complex, 
    # so we just recompute for simplicity if N is small (50).
    # N=50, conv length 99. Very fast.
    
    for _ in range(max_iter):
        improved = False
        # Try flipping each bit
        for i in range(n):
            # Flip bit i
            old_val = best_h[i]
            best_h[i] = 1.0 - old_val
            
            new_score = _compute_ratio(best_h)
            
            if new_score > best_score:
                best_score = new_score
                improved = True
            else:
                # Revert
                best_h[i] = old_val
                
        if not improved:
            break
            
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    best_h = None
    best_score = 0.0
    
    # 1. Try a few random binary initializations
    n_restarts = 10
    
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        # Random binary sequence
        h = np.random.choice([0.0, 1.0], size=N)
        
        # Ensure at least one 1
        if np.sum(h) == 0:
            h[0] = 1.0
            
        # Local search
        h_opt, score = _local_search_binary(h, max_iter=2000)
        
        if score > best_score:
            best_score = score
            best_h = h_opt.copy()
            
    # 2. Try to refine with continuous values?
    # For now, binary is a strong candidate. 
    # If we have time, we could try optimizing h_i in [0,1] using gradient descent.
    # But binary local search is robust.
    
    if best_h is None:
        return [1.0] * N
        
    return list(best_h)
