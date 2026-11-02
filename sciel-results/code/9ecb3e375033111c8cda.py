"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    
    # Attempt to construct a rank 64 decomposition
    # R(3,3,8) <= 2 * R(3,3,4)
    # R(3,3,4) <= R(3,3,3) + R(3,3,1)
    # R(3,3,3) = 23 (Winograd)
    # R(3,3,1) = 9 (Schoolbook)
    # Total Rank = 2 * (23 + 9) = 64
    
    # We define the decomposition for 3x3x3 (Rank 23)
    # and 3x3x1 (Rank 9)
    
    # For 3x3x1, it's just schoolbook:
    # C_i = sum_j A_ij B_j
    # Terms: (i, j) for i in 0..2, j in 0..2.
    
    # For 3x3x3, we need the 23 terms.
    # I will use a known set of 23 terms for 3x3 matrix multiplication.
    # If I cannot guarantee correctness, I will use schoolbook (27 terms) for 3x3x3.
    # R(3,3,4) = 27 + 9 = 36.
    # R(3,3,8) = 72.
    
    # Since I don't have the 23 terms memorized and verified,
    # I will use the schoolbook decomposition for the entire 3x3x8 problem.
    # Rank 72.
    
    # To potentially improve, I can try to use Strassen for 2x2x2 parts.
    # But 3x3 is not 2x2.
    
    # I will return the schoolbook decomposition.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
