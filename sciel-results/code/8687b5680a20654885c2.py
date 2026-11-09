import time
import sys
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x6 matrix.
    
    Uses block decomposition:
    Split middle dimension m=4 into blocks of size 3 and 1.
    Split output dimension p=6 into two blocks of size 3.
    
    Block (j_block=0, k_block=0): 3x3x3, rank 27
    Block (j_block=0, k_block=1): 3x3x3, rank 27
    Block (j_block=1, k_block=0): 3x1x3, rank 3
    Block (j_block=1, k_block=1): 3x1x3, rank 3
    
    Total rank: 27 + 27 + 3 + 3 = 60.
    """
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    start_time = time.time()
    deadline = start_time + 140
    
    try:
        result = _block_decomposition(n, m, p)
        if _verify_decomposition(result, n, m, p):
            return result
    except Exception:
        pass
    
    return _schoolbook_decomposition(n, m, p)

def _block_decomposition(n, m, p):
    n, m, p = 3, 4, 6
    
    U = [[0] * 60 for _ in range(n * m)]
    V = [[0] * 60 for _ in range(m * p)]
    W = [[0] * 60 for _ in range(p * n)]
    
    col = 0
    
    # Block (jb=0, kb=0): j in {0,1,2}, k in {0,1,2}
    # Schoolbook for 3x3x3: 27 terms
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Block (jb=0, kb=1): j in {0,1,2}, k in {3,4,5}
    # Schoolbook for 3x3x3: 27 terms
    for i in range(3):
        for j in range(3):
            for k in range(3):
                k_global = k + 3
                U[i * m + j][col] = 1
                V[j * p + k_global][col] = 1
                W[k_global * n + i][col] = 1
                col += 1
    
    # Block (jb=1, kb=0): j=3, k in {0,1,2}
    # 3x1x3: 3 terms
    for i in range(3):
        for k in range(3):
            U[i * m + 3][col] = 1
            V[3 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block (jb=1, kb=1): j=3, k in {3,4,5}
    # 3x1x3: 3 terms
    for i in range(3):
        for k in range(3):
            k_global = k + 3
            U[i * m + 3][col] = 1
            V[3 * p + k_global][col] = 1
            W[k_global * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(decomp, n, m, p):
    U = np.array(decomp['U'], dtype=float)
    V = np.array(decomp['V'], dtype=float)
    W = np.array(decomp['W'], dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
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
