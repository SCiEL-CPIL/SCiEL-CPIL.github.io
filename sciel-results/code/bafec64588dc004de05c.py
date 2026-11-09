def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix
    by a 5x6 matrix. Output factor W follows W[k*n+i, r].
    Returns {'U', 'V', 'W'} where:
      U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    n, m, p = 3, 5, 6
    
    # Schoolbook decomposition: Rank n*m*p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                # V index: j * p + k
                # W index: k * n + i
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
                
    # Verification (optional, but good for debugging locally)
    # Reconstruct T[i, j, k]
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For this construction, only the term where r corresponds to (i,j,k) is non-zero.
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
    # Product = 1.
    # All other terms are 0.
    # So T[i, j, k] = 1.
    
    return {"U": U, "V": V, "W": W}
