<<<<<<< SEARCH
def construct():
    n, m, p = DIMENSIONS
    
    # Try to find rank 16
    # Given the difficulty, we will return the known rank 17 solution.
    # If a rank 16 solution were easily available, we would use it.
    # For now, we ensure the rank 17 solution is valid and fast.
    
    U, V, W = _build_rank17()
    
    # Verify just in case
    # if not _verify_decomposition(U, V, W, n, m, p):
    #     # Fallback to schoolbook rank 18 if something goes wrong (should not happen)
    #     pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to find rank 16
    result16 = _try_rank16_search()
    
    if result16 is not None:
        U, V, W = result16
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 17
    U, V, W = _build_rank17()
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # This should not happen, but handle gracefully
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE
