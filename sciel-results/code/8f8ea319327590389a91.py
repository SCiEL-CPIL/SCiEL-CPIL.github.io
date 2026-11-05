import numpy as np

def construct():
    n, m, p = 3, 3, 7
    # Schoolbook decomposition for matrix multiplication of (n x m) * (m x p)
    # T[i, j, k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) of C
    # Using trace convention: W[k*n + i, r]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # For each (i, j, k), we need a rank-1 term where:
    #   U[i*m + j, r] = 1
    #   V[j*p + k, r] = 1
    #   W[k*n + i, r] = 1
    
    rank = n * m * p  # 63
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
