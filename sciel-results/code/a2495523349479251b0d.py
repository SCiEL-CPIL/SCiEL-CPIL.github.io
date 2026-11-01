"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 16 decomposition using c indexed by (i, k)
    rank = n * p  # 16
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i0 in range(n):
        for k0 in range(p):
            # U[i, j, c] = 1/2 if i == i0
            for i in range(n):
                for j in range(m):
                    if i == i0:
                        U[i * m + j][col] = 0.5
            # V[j, k, c] = 1 if k == k0
            for j in range(m):
                for k in range(p):
                    if k == k0:
                        V[j * p + k][col] = 1.0
            # W[k, i, c] = 1 if k == k0 and i == i0 (W indexed by k*n+i)
            for k in range(p):
                for i in range(n):
                    if k == k0 and i == i0:
                        W[k * n + i][col] = 1.0
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
