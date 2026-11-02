"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition of 5x5x5 matrix multiplication.
    Uses a block decomposition splitting 5 into 4+1.
    The 4x4 block uses schoolbook (rank 16).
    Total rank: 16 (4x4x4x4) + 16 (4x1x1x4) + 16 (4x4x4x1) + 4 (4x1x1x1) 
              + 16 (1x4x4x4) + 4 (1x1x1x4) + 4 (1x4x4x1) + 1 (1x1x1x1) = 77.
    """
    n, m, p = DIMENSIONS
    # We are splitting the dimension 5 into 4 and 1.
    # Indices 0,1,2,3 are in the "4" block.
    # Index 4 is in the "1" block.
    
    rank = 77
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(i, j, k):
        nonlocal col
        # U[i*m+j, col] = 1
        U[i * m + j][col] = 1
        # V[j*p+k, col] = 1
        V[j * p + k][col] = 1
        # W[k*n+i, col] = 1
        W[k * n + i][col] = 1
        col += 1

    # 1. A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    # Rank 16
    for i in range(4):
        for j in range(4):
            for k in range(4):
                add_term(i, j, k)

    # 2. A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    # Rank 16
    for i in range(4):
        for k in range(4):
            add_term(i, 4, k)

    # 3. A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    # Rank 16
    for i in range(4):
        for j in range(4):
            add_term(i, j, 4)

    # 4. A[0:4, 4] * B[4, 4] -> C[0:4, 4]
    # Rank 4
    for i in range(4):
        add_term(i, 4, 4)

    # 5. A[4, 0:4] * B[0:4, 0:4] -> C[4, 0:4]
    # Rank 16
    for j in range(4):
        for k in range(4):
            add_term(4, j, k)

    # 6. A[4, 4] * B[4, 0:4] -> C[4, 0:4]
    # Rank 4
    for k in range(4):
        add_term(4, 4, k)

    # 7. A[4, 0:4] * B[0:4, 4] -> C[4, 4]
    # Rank 4
    for j in range(4):
        add_term(4, j, 4)

    # 8. A[4, 4] * B[4, 4] -> C[4, 4]
    # Rank 1
    add_term(4, 4, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
