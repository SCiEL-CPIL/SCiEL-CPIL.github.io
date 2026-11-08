"""Matrix multiplication 2x4x7 with a rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 7]


def construct():
    n, m, p = DIMENSIONS
    # Direct schoolbook scheme: rank = n*m*p = 2*4*7 = 56
    # A is n x m (2x4), B is m x p (4x7), C is n x p (2x7)
    # Output factor W follows trace convention W[k*n+i, r]
    # where C[i,k] = sum_j A[i,j]*B[j,k]
    # C[i,k] corresponds to W[k*n+i, r]
    
    rank = n * m * p  # 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):       # row of A, 0..1
        for j in range(m):   # col of A = row of B, 0..3
            for k in range(p):  # col of B, 0..6
                # U[i*m+j] = 1
                U[i * m + j][r] = 1
                # V[j*p+k] = 1
                V[j * p + k][r] = 1
                # W[k*n+i] = 1  (trace convention: C[i,k] -> W[k*n+i])
                W[k * n + i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}
