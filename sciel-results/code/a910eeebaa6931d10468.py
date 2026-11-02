"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_rank53():
    """
    Guaranteed rank-53 decomposition using Strassen for 7 blocks of 2 columns
    plus schoolbook for the last column.
    7 * 7 + 4 = 53
    """
    n, m, p = DIMENSIONS
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    U_school = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    V_school = [[1,0,1,0],[0,1,0,1]]
    W_school = [[1,1,0,0],[0,0,1,1]]
    
    col_offset = 0
    for block in range(7):
        k0 = 2 * block
        k1 = 2 * block + 1
        for r in range(7):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = U_strassen[i * 2 + j][r]
            for j in range(2):
                V[j * p + k0][c] = V_strassen[j * 2][r]
                V[j * p + k1][c] = V_strassen[j * 2 + 1][r]
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk * n + i][c] = W_strassen[k * 2 + i][r]
        col_offset += 7
    
    k_last = 14
    for r in range(4):
        c = col_offset + r
        for i in range(2):
            for j in range(2):
                U[i * m + j][c] = U_school[i * 2 + j][r]
        for j in range(2):
            V[j * p + k_last][c] = V_school[j][r]
        for i in range(2):
            W[k_last * n + i][c] = W_school[i][r]
    
    return {"U": U, "V": V, "W": W}

def _build_rank48():
    """
    Guaranteed rank-48 decomposition using Strassen for 6 blocks of 2 columns
    plus schoolbook for the last 3 columns.
    6 * 7 + 3 * 4 = 42 + 12 = 54... wait that's 54.
    
    Actually let me reconsider. 7 blocks of 2 cols = 14 cols (rank 49) + 1 col (rank 4) = 53.
    6 blocks of 2 cols = 12 cols (rank 42) + 3 cols (rank 12) = 54. That's worse.
    
    Let me try: 7 blocks of 2 cols (rank 49) + 1 col (rank 4) = 53.
    Or: 6 blocks of 2 cols (rank 42) + 3 cols via Strassen-like (rank 7) = 49? No, 3 cols isn't a power of 2.
    
    Actually, the best guaranteed construction is still rank 53.
    Let me try a different approach: use 2x3 blocks.
    
    For a 2x2 matrix times 2x3 matrix, can we do better than 4+4+4=12 (schoolbook) or 2*2*3=12?
    Strassen gives rank 7 for 2x2x2. For 2x2x3, we can do Strassen on the first 2 cols (rank 7) + schoolbook on last col (rank 4) = 11.
    Or we could try to find a rank-10 decomposition.
    
    Let me use 5 blocks of 2x2x3: 5 * 11 = 55. Worse.
    
    Best is still 7*7 + 4 = 53.
    """
    return _build_rank53()

def construct():
    """
    Entry point: return the best known decomposition.
    Currently returns the guaranteed rank-53 decomposition.
    """
    n, m, p = DIMENSIONS
    return _build_rank53()
# EVOLVE-BLOCK-END
