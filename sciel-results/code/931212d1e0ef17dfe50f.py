import time

DIMENSIONS = [3, 5, 6]

def construct():
    """
    Returns {'U', 'V', 'W'} with integer entries that exactly decompose
    the 3x5 @ 5x6 matrix multiplication tensor.
    
    Strategy:
    Split A (3x5) into A_top (2x5) and A_bottom (1x5).
    Split B (5x6) into B_top (2x6) and B_bottom (3x6) based on inner dim split 2+3.
    
    C = A_top @ B_top + A_top @ B_bottom + A_bottom @ B_top + A_bottom @ B_bottom
    
    We further split the output columns of B (and thus C) into three 2-column blocks.
    
    Total Rank = 84.
    """
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Total rank calculation:
    # Part 1 (A_top 2x5):
    #   1a (2x2 @ 2x6): 3 blocks of 2x2x2 (Strassen rank 7) -> 21
    #   1b (2x3 @ 3x6): 3 blocks of 2x3x2 (Decomp rank 11) -> 33
    # Part 2 (A_bottom 1x5):
    #   2a (1x2 @ 2x6): 3 blocks of 1x2x2 (Schoolbook rank 4) -> 12
    #   2b (1x3 @ 3x6): 3 blocks of 1x3x2 (Schoolbook rank 6) -> 18
    # Total: 21 + 33 + 12 + 18 = 84
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 decomposition
    # u_offset: index in U for A[0,0]
    # v_offset: index in V for B[0,0]
    # w_offset: index in W for C[0,0]
    # u_stride_row, u_stride_col: strides for A indices
    # v_stride_row, v_stride_col: strides for B indices
    # w_stride_row, w_stride_col: strides for C indices
    
    def add_strassen_2x2(u_off, v_off, w_off, u_sr, u_sc, v_sr, v_sc, w_sr, w_sc):
        nonlocal col
        if time.time() > deadline:
            return
        
        # M1 = (A00+A11)(B00+B11)
        U[u_off + 0*u_sr + 0*u_sc][col] = 1
        U[u_off + 1*u_sr + 1*u_sc][col] = 1
        V[v_off + 0*v_sr + 0*v_sc][col] = 1
        V[v_off + 1*v_sr + 1*v_sc][col] = 1
        W[w_off + 0*w_sr + 0*w_sc][col] = 1
        W[w_off + 0*w_sr + 1*w_sc][col] = 1
        W[w_off + 1*w_sr + 0*w_sc][col] = 1
        W[w_off + 1*w_sr + 1*w_sc][col] = -1
        col += 1
        
        # M2 = (A10+A11)B00
        U[u_off + 1*u_sr + 0*u_sc][col] = 1
        U[u_off + 1*u_sr + 1*u_sc][col] = 1
        V[v_off + 0*v_sr + 0*v_sc][col] = 1
        W[w_off + 1*w_sr + 0*w_sc][col] = 1
        W[w_off + 1*w_sr + 1*w_sc][col] = -1
        col += 1
        
        # M3 = A00(B01-B11)
        U[u_off + 0*u_sr + 0*u_sc][col] = 1
        V[v_off + 0*v_sr + 1*v_sc][col] = 1
        V[v_off + 1*v_sr + 1*v_sc][col] = -1
        W[w_off + 0*w_sr + 1*w_sc][col] = 1
        W[w_off + 1*w_sr + 1*w_sc][col] = 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[u_off + 1*u_sr + 1*u_sc][col] = 1
        V[v_off + 1*v_sr + 0*v_sc][col] = 1
        V[v_off + 0*v_sr + 0*v_sc][col] = -1
        W[w_off + 0*w_sr + 0*w_sc][col] = 1
        W[w_off + 1*w_sr + 0*w_sc][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[u_off + 0*u_sr + 0*u_sc][col] = 1
        U[u_off + 0*u_sr + 1*u_sc][col] = 1
        V[v_off + 1*v_sr + 1*v_sc][col] = 1
        W[w_off + 0*w_sr + 0*w_sc][col] = -1
        W[w_off + 1*w_sr + 1*w_sc][col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[u_off + 1*u_sr + 0*u_sc][col] = 1
        U[u_off + 0*u_sr + 0*u_sc][col] = -1
        V[v_off + 0*v_sr + 0*v_sc][col] = 1
        V[v_off + 0*v_sr + 1*v_sc][col] = 1
        W[w_off + 1*w_sr + 1*w_sc][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[u_off + 0*u_sr + 1*u_sc][col] = 1
        U[u_off + 1*u_sr + 1*u_sc][col] = -1
        V[v_off + 1*v_sr + 0*v_sc][col] = 1
        V[v_off + 1*v_sr + 1*v_sc][col] = 1
        W[w_off + 0*w_sr + 0*w_sc][col] = 1
        W[w_off + 1*w_sr + 1*w_sc][col] = 1
        col += 1

    def add_schoolbook_2x2x2(u_off, v_off, w_off, u_sr, u_sc, v_sr, v_sc, w_sr, w_sc):
        nonlocal col
        if time.time() > deadline:
            return
        for i in range(2):
            for k in range(2):
                for j in range(2):
                    U[u_off + i*u_sr + k*u_sc][col] = 1
                    V[v_off + k*v_sr + j*v_sc][col] = 1
                    W[w_off + i*w_sr + j*w_sc][col] = 1
                    col += 1

    def add_schoolbook_1x2x2(u_off, v_off, w_off, u_sr, u_sc, v_sr, v_sc, w_sr, w_sc):
        nonlocal col
        if time.time() > deadline:
            return
        # A is 1x2, B is 2x2, C is 1x2
        for k in range(2):
            for j in range(2):
                U[u_off + 0*u_sr + k*u_sc][col] = 1
                V[v_off + k*v_sr + j*v_sc][col] = 1
                W[w_off + 0*w_sr + j*w_sc][col] = 1
                col += 1

    def add_schoolbook_1x3x2(u_off, v_off, w_off, u_sr, u_sc, v_sr, v_sc, w_sr, w_sc):
        nonlocal col
        if time.time() > deadline:
            return
        # A is 1x3, B is 3x2, C is 1x2
        for k in range(3):
            for j in range(2):
                U[u_off + 0*u_sr + k*u_sc][col] = 1
                V[v_off + k*v_sr + j*v_sc][col] = 1
                W[w_off + 0*w_sr + j*w_sc][col] = 1
                col += 1

    # Indices in global matrices:
    # U index: i * m + k  (i: 0..2, k: 0..4)
    # V index: k * p + j  (k: 0..4, j: 0..5)
    # W index: j * n + i  (j: 0..5, i: 0..2)  <-- Note: j is col, i is row
    
    # Strides for local 2x2 blocks in global space:
    # If local A is at global row r, col c:
    # u_off = r * m + c
    # u_sr = m (5)
    # u_sc = 1
    
    # If local B is at global row r, col c:
    # v_off = r * p + c
    # v_sr = p (6)
    # v_sc = 1
    
    # If local C is at global row r, col c:
    # w_off = c * n + r
    # w_sr = 1 (row changes index by 1)
    # w_sc = n (3) (col changes index by 3)
    
    # Part 1a: A[0:2, 0:2] * B[0:2, 0:6]
    # A rows 0,1; cols 0,1. B rows 0,1; cols 0,5.
    # Split B cols into 3 blocks of 2: [0:2], [2:4], [4:6]
    
    for block_idx in range(3):
        b_col_start = block_idx * 2
        # A is fixed at rows 0,1 cols 0,1
        u_off = 0 * m + 0
        # B is rows 0,1 cols b_col_start, b_col_start+1
        v_off = 0 * p + b_col_start
        # C is rows 0,1 cols b_col_start, b_col_start+1
        w_off = b_col_start * n + 0
        
        add_strassen_2x2(u_off, v_off, w_off, m, 1, p, 1, 1, n)
        if time.time() > deadline: break

    # Part 1b: A[0:2, 2:5] * B[2:5, 0:6]
    # A rows 0,1; cols 2,3,4. B rows 2,3,4; cols 0,5.
    # Split B cols into 3 blocks of 2: [0:2], [2:4], [4:6]
    
    for block_idx in range(3):
        if time.time() > deadline: break
        b_col_start = block_idx * 2
        
        # Decompose 2x3x2 into 2x1x2 (schoolbook) + 2x2x2 (strassen)
        # Split inner dim 3 into 1 and 2.
        # A_part1: rows 0,1; col 2. (1 col)
        # B_part1: row 2; cols b_col_start, b_col_start+1.
        # C_part1: rows 0,1; cols b_col_start, b_col_start+1.
        
        # 2x1x2 Schoolbook
        u_off_1 = 0 * m + 2
        v_off_1 = 2 * p + b_col_start
        w_off_1 = b_col_start * n + 0
        
        # U: 2 rows, 1 col. Strides: row=5, col=1
        # V: 1 row, 2 cols. Strides: row=6, col=1
        # W: 2 rows, 2 cols. Strides: row=1, col=3
        for i in range(2):
            for j in range(2):
                U[u_off_1 + i*m + 0][col] = 1
                V[v_off_1 + 0*p + j][col] = 1
                W[w_off_1 + i*1 + j*n][col] = 1
                col += 1
                if time.time() > deadline: break
            if time.time() > deadline: break
        if time.time() > deadline: break

        # A_part2: rows 0,1; cols 3,4. (2 cols)
        # B_part2: rows 3,4; cols b_col_start, b_col_start+1.
        # C_part2: rows 0,1; cols b_col_start, b_col_start+1.
        
        u_off_2 = 0 * m + 3
        v_off_2 = 3 * p + b_col_start
        w_off_2 = b_col_start * n + 0
        
        add_strassen_2x2(u_off_2, v_off_2, w_off_2, m, 1, p, 1, 1, n)
        if time.time() > deadline: break

    # Part 2a: A[2:3, 0:2] * B[0:2, 0:6]
    # A row 2; cols 0,1. B rows 0,1; cols 0,5.
    # Split B cols into 3 blocks of 2.
    
    for block_idx in range(3):
        if time.time() > deadline: break
        b_col_start = block_idx * 2
        
        u_off = 2 * m + 0
        v_off = 0 * p + b_col_start
        w_off = b_col_start * n + 2
        
        # 1x2x2 Schoolbook
        for k in range(2):
            for j in range(2):
                U[u_off + 0*m + k][col] = 1
                V[v_off + k*p + j][col] = 1
                W[w_off + 0*1 + j*n][col] = 1
                col += 1
                if time.time() > deadline: break
            if time.time() > deadline: break
        if time.time() > deadline: break

    # Part 2b: A[2:3, 2:5] * B[2:5, 0:6]
    # A row 2; cols 2,3,4. B rows 2,3,4; cols 0,5.
    # Split B cols into 3 blocks of 2.
    
    for block_idx in range(3):
        if time.time() > deadline: break
        b_col_start = block_idx * 2
        
        u_off = 2 * m + 2
        v_off = 2 * p + b_col_start
        w_off = b_col_start * n + 2
        
        # 1x3x2 Schoolbook
        for k in range(3):
            for j in range(2):
                U[u_off + 0*m + k][col] = 1
                V[v_off + k*p + j][col] = 1
                W[w_off + 0*1 + j*n][col] = 1
                col += 1
                if time.time() > deadline: break
            if time.time() > deadline: break
        if time.time() > deadline: break

    return {"U": U, "V": V, "W": W}
