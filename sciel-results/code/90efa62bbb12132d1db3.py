<<<<<<< SEARCH
def construct():
    """
    Constructs the schoolbook tensor decomposition for matrix multiplication.
    For n=2, m=3, p=10, the rank is n*m*p = 60.
    This is the baseline decomposition.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best_result = _schoolbook()
    best_rank = len(best_result["U"][0])
    
    # Verify the schoolbook decomposition
    if not _verify_decomposition(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        # Should not happen, but if it does, we have a problem
        pass
    
    # Try to find a lower rank decomposition
    # For 2x3x10 matrix multiplication, we can try various approaches
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # But the actual minimal rank is unknown; let's try to find something better than 60
    
    # Strategy: Use the fact that for small dimensions, we can sometimes find
    # decompositions with rank less than n*m*p by exploiting structure
    
    # For now, return the schoolbook decomposition which is guaranteed correct
    # A more sophisticated search would try to find linear dependencies
    
    elapsed = time.time() - start_time
    if elapsed > 10:
        print(f"Warning: construct took {elapsed}s")
    
    return best_result
=======
def construct():
    """
    Constructs the rank-1 tensor decomposition for matrix multiplication.
    For n=2, m=3, p=10, the tensor T[i,j,k] = 1 for all i,j,k.
    The all-ones tensor has rank 1: U[i*m+j,0]=1, V[j*p+k,0]=1, W[k*n+i,0]=1.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank = 1
    
    # Rank-1 decomposition: all entries are 1
    U = [[1.0] * rank for _ in range(n * m)]
    V = [[1.0] * rank for _ in range(m * p)]
    W = [[1.0] * rank for _ in range(p * n)]
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails (should not happen)
        return _schoolbook()
    
    elapsed = time.time() - start_time
    if elapsed > 10:
        print(f"Warning: construct took {elapsed}s")
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
