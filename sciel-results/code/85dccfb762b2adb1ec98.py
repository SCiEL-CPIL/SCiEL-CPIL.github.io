"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is m * p
    rank = m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            
            # U: indices (i, j)
            # U[i*m + j, c] = 1 for all i
            for i in range(n):
                U[i * m + j][c] = 1
                
            # V: indices (j, k)
            # V[j*p + k, c] = 1
            V[j * p + k][c] = 1
            
            # W: indices (k, i)
            # W[k*n + i, c] = 1 for all i
            for i in range(n):
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
