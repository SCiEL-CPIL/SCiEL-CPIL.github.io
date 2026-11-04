import numpy as np

def construct():
    n, m, p = 4, 4, 8
    rank = 128  # schoolbook: 4*4*8 = 128 terms
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Schoolbook decomposition for 4x4x8
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..3, k in 0..7
    # Each term (i,j,k) is rank-1:
    #   U[i*m+j] = 1
    #   V[j*p+k] = 1
    #   W[k*n+i] = 1
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
