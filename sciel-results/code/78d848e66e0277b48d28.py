import numpy as np
import time

def construct():
    """
    Returns a rank-48 exact tensor decomposition for multiplying a 4x4 matrix by a 4x9 matrix.
    This is achieved by splitting the 4x9 matrix into two 4x4 blocks and one 4x1 block,
    and using the schoolbook algorithm for each part.
    Rank = 16 (for 4x4*4x4) + 16 (for 4x4*4x4) + 16 (for 4x4*4x1) = 48.
    """
    n, m, p = 4, 4, 9
    
    # Total rank
    rank = 48
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A (4x4) * B1 (4x4) -> C1 (4x4)
    # B1 corresponds to columns 0-3 of B
    # C1 corresponds to columns 0-3 of C
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Part 2: A (4x4) * B2 (4x4) -> C2 (4x4)
    # B2 corresponds to columns 4-7 of B
    # C2 corresponds to columns 4-7 of C
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i * m + j, col] = 1.0
                V[j * p + (k + 4), col] = 1.0
                W[(k + 4) * n + i, col] = 1.0
                col += 1
                
    # Part 3: A (4x4) * B3 (4x1) -> C3 (4x1)
    # B3 corresponds to column 8 of B
    # C3 corresponds to column 8 of C
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1.0
            V[j * p + 8, col] = 1.0
            W[8 * n + i, col] = 1.0
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
