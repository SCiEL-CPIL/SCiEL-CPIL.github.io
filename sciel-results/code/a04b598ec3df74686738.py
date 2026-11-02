"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    Uses 5 independent Strassen rank-7 blocks.
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 5
    
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    rank = num_blocks * 7  # 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(num_blocks):
        k0 = 2 * b
        k1 = 2 * b + 1
        for r in range(7):
            col = b * 7 + r
            U[0][col] = strassen_U[r][0]
            U[1][col] = strassen_U[r][1]
            U[2][col] = strassen_U[r][2]
            U[3][col] = strassen_U[r][3]
            
            V[0 * p + k0][col] = strassen_V[r][0]
            V[0 * p + k1][col] = strassen_V[r][1]
            V[1 * p + k0][col] = strassen_V[r][2]
            V[1 * p + k1][col] = strassen_V[r][3]
            
            W[k0 * n + 0][col] = strassen_W[r][0]
            W[k1 * n + 0][col] = strassen_W[r][1]
            W[k0 * n + 1][col] = strassen_W[r][2]
            W[k1 * n + 1][col] = strassen_W[r][3]
    
    # Verify
    def verify(U_mat, V_mat, W_mat, n, m, p):
        rank = len(U_mat[0]) if U_mat else 0
        for i in range(n):
            for k in range(p):
                total = 0
                for j in range(m):
                    for r in range(rank):
                        total += U_mat[i * m + j][r] * V_mat[j * p + k][r] * W_mat[k * n + i][r]
                if total != m:
                    return False
        return True
    
    assert verify(U, V, W, n, m, p), "Verification failed"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
