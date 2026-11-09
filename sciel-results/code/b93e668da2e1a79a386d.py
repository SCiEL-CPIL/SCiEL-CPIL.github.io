import time

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit
    
    n, m, p = 2, 4, 5
    
    # Strassen decomposition for 2x2x2
    # U, V, W are 7x4 matrices (flattened 2x2 blocks)
    # Indices: i, j in {0,1}
    # U[i*2+j]
    strassen_U = [
        [1,0,0,1], [0,0,1,1], [1,0,0,0], [0,0,0,1],
        [1,1,0,0], [-1,0,1,0], [0,1,0,-1]
    ]
    strassen_V = [
        [1,0,0,1], [1,0,0,0], [0,1,0,-1], [-1,0,1,0],
        [0,0,0,1], [1,1,0,0], [0,0,1,1]
    ]
    strassen_W = [
        [1,0,0,1], [0,1,0,-1], [0,0,1,1], [1,1,0,0],
        [-1,0,1,0], [0,0,0,1], [1,0,0,0]
    ]
    
    # Rank 30 decomposition
    # 4 Strassen blocks for 2x4x4 part (rank 28)
    # 1 combined block for 2x4x1 part (rank 2)
    rank = 28 + 2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Decompose 2x4x4 part using 4 Strassen blocks
    # Blocks: (A_cols, B_rows, C_cols)
    # 1. A[:,0:1] * B[0:1,0:2] -> C[:,0:2]
    # 2. A[:,2:3] * B[2:3,0:2] -> C[:,0:2]
    # 3. A[:,0:1] * B[0:1,2:4] -> C[:,2:4]
    # 4. A[:,2:3] * B[2:3,2:4] -> C[:,2:4]
    
    blocks = [
        (0, 0, 0), # A_col_offset, B_row_offset, C_col_offset
        (2, 2, 0),
        (0, 0, 2),
        (2, 2, 2)
    ]
    
    for a_off, b_off, c_off in blocks:
        for r in range(7):
            # U part: 2x2 block at rows 0-1, cols a_off-a_off+1
            for li in range(2):
                for lj in range(2):
                    gi = li
                    gj = a_off + lj
                    U[gi * m + gj][col] = strassen_U[r][li * 2 + lj]
            # V part: 2x2 block at rows b_off-b_off+1, cols c_off-c_off+1
            for lj in range(2):
                for lk in range(2):
                    gj = b_off + lj
                    gk = c_off + lk
                    V[gj * p + gk][col] = strassen_V[r][lj * 2 + lk]
            # W part: 2x2 block at rows c_off-c_off+1, cols 0-1
            for lk in range(2):
                for li in range(2):
                    gk = c_off + lk
                    gi = li
                    W[gk * n + gi][col] = strassen_W[r][lk * 2 + li]
            col += 1
            
    # Decompose 2x4x1 part (last column of B)
    # C[:, 4] = A[:, 0:1] * B[0:1, 4] + A[:, 2:3] * B[2:3, 4]
    # Combined rank 2
    # Term 1: Covers A[:, 0:1] and B[0:1, 4]
    # U: rows 0-1, cols 0-1 are 1
    # V: rows 0-1, col 4 is 1
    # W: row 4, cols 0-1 are 1
    for li in range(2):
        for lj in range(2):
            U[li * m + lj][col] = 1
    for lj in range(2):
        V[lj * p + 4][col] = 1
    for li in range(2):
        W[4 * n + li][col] = 1
    col += 1
    
    # Term 2: Covers A[:, 2:3] and B[2:3, 4]
    # U: rows 0-1, cols 2-3 are 1
    # V: rows 2-3, col 4 is 1
    # W: row 4, cols 0-1 are 1
    for li in range(2):
        for lj in range(2):
            U[li * m + 2 + lj][col] = 1
    for lj in range(2):
        V[(2 + lj) * p + 4][col] = 1
    for li in range(2):
        W[4 * n + li][col] = 1
    col += 1
        
    return {"U": U, "V": V, "W": W}
