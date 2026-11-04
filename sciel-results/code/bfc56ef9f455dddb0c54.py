import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # n=4 (rows of A, cols of C)
    # m=5 (cols of A, rows of B)
    # p=6 (cols of B, cols of C)
    
    # Schoolbook rank for full matrix is 120.
    # We use block decomposition.
    # Split A rows: 3, 1
    # Split B cols: 3, 3
    
    # Blocks:
    # 1. A[0:3, :], B[:, 0:3] -> C[0:3, 0:3] (3x3)
    # 2. A[0:3, :], B[:, 3:6] -> C[0:3, 3:6] (3x3)
    # 3. A[3:4, :], B[:, 0:3] -> C[3:4, 0:3] (1x3)
    # 4. A[3:4, :], B[:, 3:6] -> C[3:4, 3:6] (1x3)
    
    # Ranks:
    # 3x3 Schoolbook: 9
    # 3x3 Schoolbook: 9
    # 1x3 Schoolbook: 3
    # 1x3 Schoolbook: 3
    # Total Rank: 24
    
    R = 24
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    col_offset = 0
    
    def add_schoolbook_block(a_row_start, a_row_end, b_col_start, b_col_end, c_row_start, c_col_start):
        nonlocal col_offset
        a_rows = range(a_row_start, a_row_end)
        b_cols = range(b_col_start, b_col_end)
        # C rows correspond to A rows
        # C cols correspond to B cols
        
        for i in a_rows:
            for j in range(m):
                for k in b_cols:
                    # U index: i*m + j
                    # V index: j*p + k
                    # W index: k*n + i (Wait, W[k*n+i, r])
                    # Task says: W[k*n+i,r]
                    # k is col of C (0..p-1)
                    # i is row of C (0..n-1)
                    
                    U[i * m + j, col_offset] = 1.0
                    V[j * p + k, col_offset] = 1.0
                    W[k * n + i, col_offset] = 1.0
                    col_offset += 1
                    if col_offset >= R:
                         raise Exception("Rank overflow")

    # Block 1: A rows 0-2, B cols 0-2
    # C rows 0-2, C cols 0-2
    add_schoolbook_block(0, 3, 0, 3, 0, 0)
    
    # Block 2: A rows 0-2, B cols 3-5
    # C rows 0-2, C cols 3-5
    add_schoolbook_block(0, 3, 3, 6, 0, 3)
    
    # Block 3: A row 3, B cols 0-2
    # C row 3, C cols 0-2
    add_schoolbook_block(3, 4, 0, 3, 3, 0)
    
    # Block 4: A row 3, B cols 3-5
    # C row 3, C cols 3-5
    add_schoolbook_block(3, 4, 3, 6, 3, 3)
    
    if col_offset != R:
        # This should not happen
        pass

    return {"U": U, "V": V, "W": W}
