"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Rank 4 solution
    rank = 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for c in range(rank):
                U[i * m + j][c] = 1
                
    for j in range(m):
        for k in range(p):
            for c in range(rank):
                if k == c:
                    V[j * p + k][c] = 1
                else:
                    V[j * p + k][c] = 0
                    
    for k in range(p):
        for i in range(n):
            for c in range(rank):
                if k == c:
                    W[k * n + i][c] = 1
                else:
                    W[k * n + i][c] = 0
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
