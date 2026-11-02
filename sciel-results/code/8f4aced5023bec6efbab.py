def _build_rank1_4x1x6():
    """
    Build rank-1 decomposition for 4x1 * 1x6 -> 4x6.
    """
    n, m, p = 4, 1, 6
    rank = 1
    U = np.zeros((n * m, rank))  # (4, 1)
    V = np.zeros((m * p, rank))  # (6, 1)
    W = np.zeros((p * n, rank))  # (24, 1)
    
    for i in range(n):
        U[i * m, 0] = 1.0
    for k in range(p):
        V[k, 0] = 1.0  # j=0, so V[0*p+k, 0] = V[k, 0]
    for i in range(n):
        for k in range(p):
            W[k * n + i, 0] = 1.0
    
    return U, V, W
