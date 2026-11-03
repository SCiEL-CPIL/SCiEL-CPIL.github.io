def construct():
    t0 = time.time()
    hard_deadline = t0 + _SEARCH_TIMEOUT
    n, m, p = DIMENSIONS
    
    # Try rank 1 first: The matrix multiplication tensor is the all-ones tensor
    R = 1
    U = np.ones((n * m, R))
    V = np.ones((m * p, R))
    W = np.ones((p * n, R))
    
    if _verify_exact(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # If rank 1 fails (should not), proceed with search
    best_result = None
    
    # ... existing code ...
