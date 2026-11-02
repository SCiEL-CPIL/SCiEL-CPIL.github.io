<<<<<<< SEARCH
def construct():
    n, m, p = DIMENSIONS
    # Try the hybrid fast decomposition first; verify it exactly.
    try:
        cand = _fast3x3(n, m, p)
        if _verify(cand["U"], cand["V"], cand["W"], n, m, p):
            return cand
    except Exception:
        pass
    # Fallback to the guaranteed-correct schoolbook decomposition.
    return _schoolbook(n, m, p)
=======
def construct():
    n, m, p = DIMENSIONS
    # The tensor for matrix multiplication (with transposed output) is the all-ones tensor.
    # The rank of the all-ones tensor is 1.
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    # Verify to be safe
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to the guaranteed-correct schoolbook decomposition if rank 1 fails (should not happen)
    return _schoolbook(n, m, p)
>>>>>>> REPLACE
