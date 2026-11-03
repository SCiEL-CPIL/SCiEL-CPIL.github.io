"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    # U shape: (n*m, rank) = (20, rank)
    # V shape: (m*p, rank) = (25, rank)
    # W shape: (p*n, rank) = (20, rank)
    
    # Standard schoolbook rank: n*m*p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate over all i, j, k
    # A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Index in U for A[i,j]: i*m + j
    # Index in V for B[j,k]: j*p + k
    # Index in W for C[i,k]: k*n + i (trace convention W[k*n+i, r])
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U column r, row i*m + j
                U[i * m + j][r] = 1
                # V column r, row j*p + k
                V[j * p + k][r] = 1
                # W column r, row k*n + i
                W[k * n + i][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
