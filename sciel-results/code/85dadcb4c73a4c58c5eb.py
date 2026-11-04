"""Matrix multiplication 3x4x6 with rank-minimal exact tensor decomposition using Strassen-style block decomposition."""
DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    # Block decomposition:
    # A (3x4) split into:
    #   A_top (2x4): rows 0-1
    #   A_bot (1x4): row 2
    # A_top split into:
    #   A_top_left (2x2): cols 0-1
    #   A_top_right (2x2): cols 2-3
    # A_bot split into:
    #   A_bot_left (1x2): cols 0-1
    #   A_bot_right (1x2): cols 2-3
    #
    # B (4x6) split into:
    #   B_top (2x6): rows 0-1
    #   B_bot (2x6): rows 2-3
    # B_top split into 3 blocks of 2x2: cols 0-1, 2-3, 4-5
    # B_bot split into 3 blocks of 2x2: cols 0-1, 2-3, 4-5
    #
    # C (3x6):
    #   C_top (2x6): rows 0-1
    #   C_bot (1x6): row 2
    #
    # C_top = A_top_left * B_top + A_top_right * B_bot
    # C_bot = A_bot_left * B_top + A_bot_right * B_bot
    #
    # Each 2x2x2 block uses Strassen rank 7
    # Each 1x2x2 block uses schoolbook rank 4
    # Total: 6 * 7 + 6 * 4 = 42 + 24 = 66
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]  # 12 x 66
    V = [[0] * rank for _ in range(m * p)]  # 24 x 66
    W = [[0] * rank for _ in range(p * n)]  # 18 x 66
    
    col = 0
    
    def a_idx(i, j):
        """Index into U for A[i][j]"""
        return i * m + j
    
    def b_idx(j, k):
        """Index into V for B[j][k]"""
        return j * p + k
    
    def c_idx(i, k):
        """Index into W for C[i][k] (trace convention: W[k*n+i])"""
        return k * n + i
    
    def add_strassen_2x2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, col_start):
        """
        Add Strassen decomposition for 2x2x2 block.
        a_rows: [r0, r1], a_cols: [c0, c1] -> A entries
        b_rows: [r0, r1], b_cols: [c0, c1] -> B entries
        c_rows: [r0, r1], c_cols: [c0, c1] -> C entries
        """
        nonlocal col
        # x11 = A[a_rows[0]][a_cols[0]], x12 = A[a_rows[0]][a_cols[1]]
        # x21 = A[a_rows[1]][a_cols[0]], x22 = A[a_rows[1]][a_cols[1]]
        # y11 = B[b_rows[0]][b_cols[0]], y12 = B[b_rows[0]][b_cols[1]]
        # y21 = B[b_rows[1]][b_cols[0]], y22 = B[b_rows[1]][b_cols[1]]
        # c11 = C[c_rows[0]][c_cols[0]], c12 = C[c_rows[0]][c_cols[1]]
        # c21 = C[c_rows[1]][c_cols[0]], c22 = C[c_rows[1]][c_cols[1]]
        
        # M1 = (x11+x22)*(y11+y22)
        U[a_idx(a_rows[0], a_cols[0])][col_start] += 1
        U[a_idx(a_rows[1], a_cols[1])][col_start] += 1
        V[b_idx(b_rows[0], b_cols[0])][col_start] += 1
        V[b_idx(b_rows[1], b_cols[1])][col_start] += 1
        W[c_idx(c_rows[0], c_cols[0])][col_start] += 1
        W[c_idx(c_rows[1], c_cols[1])][col_start] += 1
        col += 1
        
        # M2 = (x21+x22)*y11
        U[a_idx(a_rows[1], a_cols[0])][col_start] += 1
        U[a_idx(a_rows[1], a_cols[1])][col_start] += 1
        V[b_idx(b_rows[0], b_cols[0])][col_start] += 1
        W[c_idx(c_rows[1], c_cols[0])][col_start] += 1
        W[c_idx(c_rows[1], c_cols[1])][col_start] += 1
        col += 1
        
        # M3 = x11*(y12-y22)
        U[a_idx(a_rows[0], a_cols[0])][col_start] += 1
        V[b_idx(b_rows[0], b_cols[1])][col_start] += 1
        V[b_idx(b_rows[1], b_cols[1])][col_start] -= 1
        W[c_idx(c_rows[0], c_cols[1])][col_start] += 1
        W[c_idx(c_rows[1], c_cols[1])][col_start] += 1
        col += 1
        
        # M4 = x22*(y21-y11)
        U[a_idx(a_rows[1], a_cols[1])][col_start] += 1
        V[b_idx(b_rows[1], b_cols[0])][col_start] += 1
        V[b_idx(b_rows[0], b_cols[0])][col_start] -= 1
        W[c_idx(c_rows[0], c_cols[0])][col_start] += 1
        W[c_idx(c_rows[1], c_cols[0])][col_start] += 1
        col += 1
        
        # M5 = (x11+x12)*y22
        U[a_idx(a_rows[0], a_cols[0])][col_start] += 1
        U[a_idx(a_rows[0], a_cols[1])][col_start] += 1
        V[b_idx(b_rows[1], b_cols[1])][col_start] += 1
        W[c_idx(c_rows[0], c_cols[0])][col_start] -= 1
        W[c_idx(c_rows[0], c_cols[1])][col_start] += 1
        col += 1
        
        # M6 = (x21-x11)*(y11+y12)
        U[a_idx(a_rows[1], a_cols[0])][col_start] += 1
        U[a_idx(a_rows[0], a_cols[0])][col_start] -= 1
        V[b_idx(b_rows[0], b_cols[0])][col_start] += 1
        V[b_idx(b_rows[0], b_cols[1])][col_start] += 1
        W[c_idx(c_rows[1], c_cols[1])][col_start] += 1
        col += 1
        
        # M7 = (x12-x22)*(y21+y22)
        U[a_idx(a_rows[0], a_cols[1])][col_start] += 1
        U[a_idx(a_rows[1], a_cols[1])][col_start] -= 1
        V[b_idx(b_rows[1], b_cols[0])][col_start] += 1
        V[b_idx(b_rows[1], b_cols[1])][col_start] += 1
        W[c_idx(c_rows[0], c_cols[0])][col_start] += 1
        W[c_idx(c_rows[0], c_cols[1])][col_start] -= 1
        col += 1
        
        return col_start + 7
    
    def add_schoolbook_1x2x2(a_row, a_cols, b_rows, b_cols, c_row, c_cols, col_start):
        """
        Add schoolbook decomposition for 1x2x2 block.
        a_row: row index in A, a_cols: [c0, c1]
        b_rows: [r0, r1], b_cols: [c0, c1]
        c_row: row index in C, c_cols: [c0, c1]
        """
        nonlocal col
        # c0 = A[a_row][a_cols[0]] * B[b_rows[0]][b_cols[0]] + A[a_row][a_cols[1]] * B[b_rows[1]][b_cols[0]]
        U[a_idx(a_row, a_cols[0])][col_start] += 1
        V[b_idx(b_rows[0], b_cols[0])][col_start] += 1
        W[c_idx(c_row, c_cols[0])][col_start] += 1
        col += 1
        
        U[a_idx(a_row, a_cols[1])][col_start] += 1
        V[b_idx(b_rows[1], b_cols[0])][col_start] += 1
        W[c_idx(c_row, c_cols[0])][col_start] += 1
        col += 1
        
        # c1 = A[a_row][a_cols[0]] * B[b_rows[0]][b_cols[1]] + A[a_row][a_cols[1]] * B[b_rows[1]][b_cols[1]]
        U[a_idx(a_row, a_cols[0])][col_start] += 1
        V[b_idx(b_rows[0], b_cols[1])][col_start] += 1
        W[c_idx(c_row, c_cols[1])][col_start] += 1
        col += 1
        
        U[a_idx(a_row, a_cols[1])][col_start] += 1
        V[b_idx(b_rows[1], b_cols[1])][col_start] += 1
        W[c_idx(c_row, c_cols[1])][col_start] += 1
        col += 1
        
        return col_start + 4
    
    # C_top = A_top_left * B_top + A_top_right * B_bot
    # A_top_left: rows 0-1, cols 0-1
    # A_top_right: rows 0-1, cols 2-3
    
    # A_top_left * B_top (3 blocks of 2x2x2)
    # Block: B_top cols 0-1
    col = add_strassen_2x2x2(
        a_rows=[0, 1], a_cols=[0, 1],
        b_rows=[0, 1], b_cols=[0, 1],
        c_rows=[0, 1], c_cols=[0, 1],
        col_start=col
    )
    # Block: B_top cols 2-3
    col = add_strassen_2x2x2(
        a_rows=[0, 1], a_cols=[0, 1],
        b_rows=[0, 1], b_cols=[2, 3],
        c_rows=[0, 1], c_cols=[2, 3],
        col_start=col
    )
    # Block: B_top cols 4-5
    col = add_strassen_2x2x2(
        a_rows=[0, 1], a_cols=[0, 1],
        b_rows=[0, 1], b_cols=[4, 5],
        c_rows=[0, 1], c_cols=[4, 5],
        col_start=col
    )
    
    # A_top_right * B_bot (3 blocks of 2x2x2)
    # Block: B_bot cols 0-1
    col = add_strassen_2x2x2(
        a_rows=[0, 1], a_cols=[2, 3],
        b_rows=[2, 3], b_cols=[0, 1],
        c_rows=[0, 1], c_cols=[0, 1],
        col_start=col
    )
    # Block: B_bot cols 2-3
    col = add_strassen_2x2x2(
        a_rows=[0, 1], a_cols=[2, 3],
        b_rows=[2, 3], b_cols=[2, 3],
        c_rows=[0, 1], c_cols=[2, 3],
        col_start=col
    )
    # Block: B_bot cols 4-5
    col = add_strassen_2x2x2(
        a_rows=[0, 1], a_cols=[2, 3],
        b_rows=[2, 3], b_cols=[4, 5],
        c_rows=[0, 1], c_cols=[4, 5],
        col_start=col
    )
    
    # C_bot = A_bot_left * B_top + A_bot_right * B_bot
    # A_bot_left: row 2, cols 0-1
    # A_bot_right: row 2, cols 2-3
    
    # A_bot_left * B_top (3 blocks of 1x2x2)
    col = add_schoolbook_1x2x2(
        a_row=2, a_cols=[0, 1],
        b_rows=[0, 1], b_cols=[0, 1],
        c_row=2, c_cols=[0, 1],
        col_start=col
    )
    col = add_schoolbook_1x2x2(
        a_row=2, a_cols=[0, 1],
        b_rows=[0, 1], b_cols=[2, 3],
        c_row=2, c_cols=[2, 3],
        col_start=col
    )
    col = add_schoolbook_1x2x2(
        a_row=2, a_cols=[0, 1],
        b_rows=[0, 1], b_cols=[4, 5],
        c_row=2, c_cols=[4, 5],
        col_start=col
    )
    
    # A_bot_right * B_bot (3 blocks of 1x2x2)
    col = add_schoolbook_1x2x2(
        a_row=2, a_cols=[2, 3],
        b_rows=[2, 3], b_cols=[0, 1],
        c_row=2, c_cols=[0, 1],
        col_start=col
    )
    col = add_schoolbook_1x2x2(
        a_row=2, a_cols=[2, 3],
        b_rows=[2, 3], b_cols=[2, 3],
        c_row=2, c_cols=[2, 3],
        col_start=col
    )
    col = add_schoolbook_1x2x2(
        a_row=2, a_cols=[2, 3],
        b_rows=[2, 3], b_cols=[4, 5],
        c_row=2, c_cols=[4, 5],
        col_start=col
    )
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
