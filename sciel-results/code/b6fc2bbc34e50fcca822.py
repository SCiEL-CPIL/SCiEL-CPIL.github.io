"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block the p dimension into two blocks of size p//2
    # This gives two independent 2x3 x 3x3 multiplications
    # Each has schoolbook rank 18, total rank 36
    # But we structure it as block operations
    
    block_p = p // 2  # 3
    rank = n * m * block_p * 2  # 18 * 2 = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for b in range(2):  # two blocks
        for i in range(n):
            for j in range(m):
                for k in range(block_p):
                    k_global = b * block_p + k
                    U[i * m + j][column] = 1
                    V[j * p + k_global][column] = 1
                    W[k_global * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
