def _construct_3x3x7():
    n, m, p = 3, 3, 7
    
    # Try rank 1: all-ones solution
    rank = 1
    U = np.ones((n*m, rank))
    V = np.ones((m*p, rank))
    W = np.ones((p*n, rank))
    
    if _verify_decomposition(U, V, W, n, m, p):
        return U, V, W, rank
    
    # Try rank 7
    rank = 7
    U = np.ones((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for t in range(rank):
        for j in range(m):
            if t < p:
                V[j*p + t, t] = 1
        for k in range(p):
            if t < p:
                for i in range(n):
                    W[k*n + i, t] = 1
    
    if _verify_decomposition(U, V, W, n, m, p):
        return U, V, W, rank
    
    # Fall back to schoolbook
    U, V, W = _schoolbook_3x3x7()
    return U, V, W, 63
