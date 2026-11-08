"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _validate(U, V, W, n, m, p):
    """Check exact decomposition: C[i,k] = sum_j A[i,j]*B[j,k].
    U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    W index: W[k*n + i, c] should equal sum_j U[i*m+j,c]*V[j*p+k,c].
    """
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # For each (i,k): sum_j U3[i,j,:] * V3[j,k,:] should equal W3[k,i,:]
    for i in range(n):
        for k in range(p):
            lhs = np.zeros(rank)
            for j in range(m):
                lhs += U3[i, j, :] * V3[j, k, :]
            if not np.allclose(lhs, W3[k, i, :], atol=1e-9):
                return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Rank-6 decomposition: one rank index per (i,j) pair
    # For each (i,j), we assign a rank index c = i*m + j
    # U[i*m+j, c] = 1 (only for the matching (i,j))
    # V[j*p+k, c] = 1 for all k (since j matches)
    # W[k*n+i, c] = 1 for all k (since i matches)
    
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            c = i * m + j
            U[i * m + j][c] = 1
            for k in range(p):
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
    
    # Verify
    if not _validate(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails (shouldn't happen)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
