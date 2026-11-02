"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Use tensor product of Strassen 2x2x2 (rank 7) with identity-like structures
    # 4x4x8 = (2x2x2) ⊗ (2x2x4)
    # Rank(2x2x2) = 7 (Strassen)
    # For 2x2x4, we can use 2 copies of 2x2x2, so rank = 14
    # Total rank for 4x4x8 = 7 * 14 = 98, which is worse than schoolbook 96
    
    # Better approach: Direct decomposition using structure
    # 3x4x8: each row of A (4 elements) multiplies B (4x8)
    # Use a rank-32 decomposition for 4x4x8 (if possible)
    # Total rank = 3 * 32 = 96 (same as schoolbook)
    
    # Let's try: 4x4x8 can be decomposed as sum of rank-1 tensors
    # Minimum known rank for 4x4x4 is 25, so 4x4x8 <= 50
    # Total: 3 * 50 = 150, worse
    
    # Use a practical approach: implement Strassen-based decomposition
    # 2x2x2 Strassen: 7 multiplications
    # 4x4x4: use 4x4 = 2x(2x2) blocks, Strassen gives 7^2 = 49 rank
    # 4x4x8: treat as 4x4x4 x 2, so rank 49*2 = 98
    
    # Since all structured approaches give >= 96, use schoolbook rank 96
    rank = n * m * p  # 96
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
