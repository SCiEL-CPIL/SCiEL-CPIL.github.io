def _build_schoolbook(n, m, p):
    """Build a valid decomposition for T_{i,j,k} = delta_{j,k}."""
    # The tensor T_{i,j,k} = 1 if j==k else 0.
    # This is equivalent to n copies of the identity matrix of size m (assuming m=p).
    # Rank is n * m.
    rank = n * m
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            # U[i*m+j, r] = 1
            U[i*m+j][r] = 1
            # V[j*p+k, r] = 1 if k == j else 0
            # Since m=p, k ranges 0..m-1.
            for k in range(p):
                if k == j:
                    V[j*p+k][r] = 1
            # W[k*n+i, r] = 1 if k == j else 0
            for k in range(p):
                if k == j:
                    W[k*n+i][r] = 1
    return U, V, W
