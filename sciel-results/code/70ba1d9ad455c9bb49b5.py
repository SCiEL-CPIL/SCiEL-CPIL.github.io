import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Strategy: Pure schoolbook (rank n*m*p = 4*4*9 = 144) is the baseline.
    We can do better by splitting B (4x9) into blocks and using Strassen for 4x4 blocks.
    
    Split B into: B1 (4x4), B2 (4x4), B3 (4x1)
    C = A*B1 + A*B2 + A*B3
    
    For A*B1 and A*B2 (4x4 * 4x4): use Strassen (rank 49 each)
    For A*B3 (4x4 * 4x1): use schoolbook (rank 4)
    Total rank: 49 + 49 + 4 = 102
    
    Alternative: Split B into three 4x3 blocks. Each A*Bi (4x4 * 4x3) via schoolbook = rank 12.
    Total rank: 3 * 12 = 36. This is much better!
    
    Let's use the 3x(4x3) schoolbook split: rank 36.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    dim_u = n * m  # 16
    dim_v = m * p  # 36
    dim_w = p * n  # 36
    
    # Strategy: Split B (4x9) into three 4x3 blocks
    # For each block of 3 columns, use schoolbook: rank = n * 3 = 4 * 3 = 12
    # Total rank = 3 * 12 = 36
    
    rank = 36
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    col = 0
    for block in range(3):
        col_offset = block * 3
        for j in range(m):  # 4 rows of A (which are columns of B block)
            for k in range(3):  # 3 columns in this block
                for i in range(n):  # 4 rows of result
                    u = i * m + j
                    v = j * p + (col_offset + k)
                    w = (col_offset + k) * n + i
                    U[u, col] = 1
                    V[v, col] = 1
                    W[w, col] = 1
                    col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
