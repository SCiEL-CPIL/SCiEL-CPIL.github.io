import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    
    # Rank 36 decomposition
    # 2 x 4 x 5 split into two 2 x 2 x 5 blocks (j=0,1 and j=2,3)
    # Each 2 x 2 x 5 split into 2 x 2 x 2 (Strassen, rank 7) + 2 x 2 x 3
    # 2 x 2 x 3 split into 2 x 2 x 2 (Strassen, rank 7) + 2 x 2 x 1 (Schoolbook, rank 4)
    # Total per 2 x 2 x 5: 7 + 7 + 4 = 18
    # Total: 18 + 18 = 36
    
    rank = 36
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for 2x2x2
    # U indices: i*2+j, V indices: j*2+k, W indices: k*2+i
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Schoolbook coefficients for 2x2x1
    SB_U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    SB_V = [
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ]
    SB_W = [
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1]
    ]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base, col_start):
        nonlocal col
        for r in range(7):
            for li in range(2):
                for lj in range(2):
                    gi = i_base + li
                    gj = j_base + lj
                    U[gi * m + gj, col_start + r] = S_U[r][li * 2 + lj]
            for lj in range(2):
                for lk in range(2):
                    gj = j_base + lj
                    gk = k_base + lk
                    V[gj * p + gk, col_start + r] = S_V[r][lj * 2 + lk]
            for lk in range(2):
                for li in range(2):
                    gk = k_base + lk
                    gi = i_base + li
                    W[gk * n + gi, col_start + r] = S_W[r][lk * 2 + li]
        col = col_start + 7
        return col

    def add_schoolbook_2x2x1_block(i_base, j_base, k_base, col_start):
        nonlocal col
        for r in range(4):
            for li in range(2):
                for lj in range(2):
                    gi = i_base + li
                    gj = j_base + lj
                    U[gi * m + gj, col_start + r] = SB_U[r][li * 2 + lj]
            for lj in range(2):
                gj = j_base + lj
                gk = k_base
                V[gj * p + gk, col_start + r] = SB_V[r][lj]
            for li in range(2):
                gk = k_base
                gi = i_base + li
                W[gk * n + gi, col_start + r] = SB_W[r][li]
        col = col_start + 4
        return col

    # Block 1: j_base = 0
    add_strassen_block(0, 0, 0, col) # k=0,1
    add_strassen_block(0, 0, 2, col) # k=2,3
    add_schoolbook_2x2x1_block(0, 0, 4, col) # k=4
    
    # Block 2: j_base = 2
    add_strassen_block(0, 2, 0, col) # k=0,1
    add_strassen_block(0, 2, 2, col) # k=2,3
    add_schoolbook_2x2x1_block(0, 2, 4, col) # k=4
    
    assert col == rank, f"Column mismatch: {col} vs {rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
