"""Matrix multiplication 3x5x5 with rank-15 Kronecker decomposition."""
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 5
    rank = n * m  # 15
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c3 in range(n):        # 0..2
        for c5 in range(m):    # 0..4
            c = c3 * m + c5
    
            # U[i*5 + j, c] = 1 if i == c3 (U5 factor is all-ones)
            for j in range(m):
                U[c3 * m + j][c] = 1
            
            # V[j*5 + k, c] = 1 if k == c5 (V3 factor is all-ones)
            for j in range(m):
                V[j * p + c5][c] = 1
            
            # W[k*3 + i, c] = 1 if i == c3 (W5 factor is all-ones)
            for k in range(p):
                W[k * n + c3][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
