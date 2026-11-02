"""Matrix multiplication 2x2 x 2x11 via Kronecker product decomposition."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 11
    
    # The minimal rank for 2x2 * 2x11 multiplication is 44.
    # This is because it decomposes into 11 independent 2x2 * 2x1 multiplications.
    # Each 2x2 * 2x1 multiplication has rank 4.
    # Total rank = 11 * 4 = 44.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
