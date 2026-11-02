def construct():
    n, m, p = DIMENSIONS
    # Rank-1 decomposition: U[i,j,0]=1, V[j,k,0]=1, W[k,i,0]=1
    # T[i,j,k] = 1*1*1 = 1 for all i,j,k
    U = np.ones((n * m, 1))
    V = np.ones((m * p, 1))
    W = np.ones((p * n, 1))
    
    # Verify
    if not _verify(U, V, W, n, m, p, tol=1e-8):
        # Fallback to schoolbook
        U, V, W = _schoolbook_global(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
