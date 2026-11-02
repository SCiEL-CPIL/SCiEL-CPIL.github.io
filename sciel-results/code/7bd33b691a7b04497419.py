"""Tensor decomposition for 3x3 by 3x6 matrix multiplication."""
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # Decompose 3x3 * 3x6 as two 3x3 * 3x3 multiplications
    # Split B = [B1, B2] where B1, B2 are 3x3
    # C = [A*B1, A*B2]
    # Each uses schoolbook rank 9, total rank 18
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]  # shape (9, 18)
    V = [[0] * rank for _ in range(m * p)]  # shape (18, 18)
    W = [[0] * rank for _ in range(p * n)]  # shape (18, 18)
    
    col = 0
    for block in range(2):
        for i in range(n):
            for j in range(m):
                for k in range(3):
                    k_global = k + 3 * block
                    U[i * m + j][col] = 1
                    V[j * p + k_global][col] = 1
                    W[k_global * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
