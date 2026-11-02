"""Program for 3x5 by 5x5 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Trivial rank: n*m*p = 75
    # Use a different enumeration: iterate k outermost, then j, then i
    # This gives the same rank but different structure
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        for j in range(m):
            for i in range(n):
                # U row index: i*m + j (row i, col j of A)
                # V row index: j*p + k (row j, col k of B)  
                # W row index: k*n + i (row k, col i of C) per trace convention
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
