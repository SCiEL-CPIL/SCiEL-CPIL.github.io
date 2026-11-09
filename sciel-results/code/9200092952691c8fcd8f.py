"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * p
    # Shapes
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill the matrices
    # c = k_prime * n + r
    for k in range(p):
        for r in range(n):
            c = k * n + r
            
            # U[i*m + j, c] = 1 if r == i else 0
            for i in range(n):
                for j in range(m):
                    if r == i:
                        U[i * m + j][c] = 1
            
            # V[j*p + k, c] = 1
            for j in range(m):
                V[j * p + k][c] = 1
            
            # W[k*n + i, c] = 1 if r == i else 0
            for i in range(n):
                if r == i:
                    W[k * n + i][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
