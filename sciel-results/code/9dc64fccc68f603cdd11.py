"""
Matrix multiplication 3x4x6 tensor decomposition using Strassen's algorithm.

Strategy:
- Split the 3x4x6 multiplication into blocks:
  A (3x4) = [A1 (3x2) | A2 (3x2)]
  B (4x6) = [B1 (2x6); B2 (2x6)]
  C = A1*B1 + A2*B2
  
- For each 3x2x6 block (A_block * B_block):
  Split B_block (2x6) into three 2x2 blocks: B_block0, B_block1, B_block2
  So C_block = A_block*B_block0 + A_block*B_block1 + A_block*B_block2
  
- For each 3x2x2 block (A_block * B_subblock):
  Split A_block (3x2) into A_top (2x2) and A_bot (1x2)
  C_top = A_top * B_subblock  (use Strassen: 7 products)
  C_bot = A_bot * B_subblock  (use schoolbook: 4 products)
  Total per 3x2x2 block: 11 products

Total rank: 2 (A blocks) * 3 (B subblocks) * 11 (products per 3x2x2) = 66
"""

def construct():
    n, m, p = 3, 4, 6
    rank = 66  # 2 * 3 * 11
    
    U = [[0] * rank for _ in range(n * m)]  # (3*4, 66) = (12, 66)
    V = [[0] * rank for _ in range(m * p)]  # (4*6, 66) = (24, 66)
    W = [[0] * rank for _ in range(p * n)]  # (6*3, 66) = (18, 66)
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1
    
    # Strassen 2x2x2: multiply (2x2) * (2x2) = (2x2)
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    # 
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # c00 = M1 - M4 + M5 + M6
    # c01 = M3 + M1 + M7
    # c10 = M2 + M4
    # c11 = M1 - M3 + M2 + M7
    #
    # Each M_k is a product of a linear form in A and a linear form in B.
    # For our factorization:
    # M_k = (sum_{i,j} u_k[i,j] * A[i,j]) * (sum_{j,k} v_k[j,k] * B[j,k])
    # and contributes to C entries via w_k[c_row, c_col]
    
    def add_strassen_2x2x2(a_row_off, b_row_off, c_row_off):
        """Add Strassen decomposition for a 2x2x2 block.
        a_row_off: offset for rows of A (2 rows)
        b_row_off: offset for rows of B (2 rows)  
        c_row_off: offset for rows of C (2 rows)
        """
        nonlocal col
        
        # M1 = (a00+a11)(b00+b11)
        # U contributions: A[a_row_off, 0] += 1, A[a_row_off+1, 1] += 1
        # V contributions: B[b_row_off, 0] += 1, B[b_row_off+1, 1] += 1
        # W contributions: C[c_row_off, 0] += 1, C[c_row_off, 1] += 1, C[c_row_off+1, 0] += 1, C[c_row_off+1, 1] += 1
        u1 = [(a_row_off, 0, 1), (a_row_off + 1, 1, 1)]
        v1 = [(b_row_off, 0, 1), (b_row_off + 1, 1, 1)]
        w1 = [(c_row_off, 0, 1), (c_row_off, 1, 1), (c_row_off + 1, 0, 1), (c_row_off + 1, 1, 1)]
        add_strassen_term(u1, v1, w1)
        
        # M2 = (a10+a11)b00
        u2 = [(a_row_off + 1, 0, 1), (a_row_off + 1, 1, 1)]
        v2 = [(b_row_off, 0, 1)]
        w2 = [(c_row_off + 1, 0, 1), (c_row_off + 1, 1, 1)]
        add_strassen_term(u2, v2, w2)
        
        # M3 = a00(b01-b11)
        u3 = [(a_row_off, 0, 1)]
        v3 = [(b_row_off, 1, 1), (b_row_off + 1, 1, -1)]
        w3 = [(c_row_off, 1, 1), (c_row_off + 1, 1, -1)]
        add_strassen_term(u3, v3, w3)
        
        # M4 = a11(b10-b00)
        u4 = [(a_row_off + 1, 1, 1)]
        v4 = [(b_row_off + 1, 0, 1), (b_row_off, 0, -1)]
        w4 = [(c_row_off, 0, -1), (c_row_off + 1, 0, 1)]
        add_strassen_term(u4, v4, w4)
        
        # M5 = (a00+a01)b11
        u5 = [(a_row_off, 0, 1), (a_row_off, 1, 1)]
        v5 = [(b_row_off + 1, 1, 1)]
        w5 = [(c_row_off, 0, 1), (c_row_off + 1, 1, 1)]
        add_strassen_term(u5, v5, w5)
        
        # M6 = (a10-a00)(b00+b01)
        u6 = [(a_row_off + 1, 0, 1), (a_row_off, 0, -1)]
        v6 = [(b_row_off, 0, 1), (b_row_off, 1, 1)]
        w6 = [(c_row_off, 0, 1)]
        add_strassen_term(u6, v6, w6)
        
        # M7 = (a01-a11)(b10+b11)
        u7 = [(a_row_off, 1, 1), (a_row_off + 1, 1, -1)]
        v7 = [(b_row_off + 1, 0, 1), (b_row_off + 1, 1, 1)]
        w7 = [(c_row_off, 1, 1), (c_row_off + 1, 1, 1)]
        add_strassen_term(u7, v7, w7)
    
    def add_strassen_term(u_list, v_list, w_list):
        """Add one rank-1 term from Strassen.
        u_list: [(row_in_A, col_in_A, val), ...] - linear form on A
        v_list: [(row_in_B, col_in_B, val), ...] - linear form on B
        w_list: [(row_in_C, col_in_C, val), ...] - contribution to C
        """
        nonlocal col
        for (ar, ac, val) in u_list:
            U[ar * m + ac][col] += val
        for (br, bc, val) in v_list:
            V[br * p + bc][col] += val
        for (cr, cc, val) in w_list:
            # W is indexed by [k*n + i, r] where k is column index of C, i is row index
            # C has shape (n, p), so C[i, k] where i in [0,n), k in [0,p)
            # W index: k * n + i
            W[cc * n + cr][col] += val
        col += 1
    
    def add_schoolbook_1x2x2(a_row, b_row_off, c_row):
        """Add schoolbook decomposition for 1x2x2 block.
        A is 1x2 (row a_row), B is 2x2 (rows b_row_off, b_row_off+1)
        C is 1x2 (row c_row)
        c0 = a0*b00 + a1*b10
        c1 = a0*b01 + a1*b11
        """
        nonlocal col
        
        # Term 1: a0 * b00 -> c0
        U[a_row * m + 0][col] = 1
        V[b_row_off * p + 0][col] = 1
        W[0 * n + c_row][col] = 1  # c0: column 0, row c_row
        col += 1
        
        # Term 2: a1 * b10 -> c0
        U[a_row * m + 1][col] = 1
        V[(b_row_off + 1) * p + 0][col] = 1
        W[0 * n + c_row][col] = 1
        col += 1
        
        # Term 3: a0 * b01 -> c1
        U[a_row * m + 0][col] = 1
        V[b_row_off * p + 1][col] = 1
        W[1 * n + c_row][col] = 1  # c1: column 1, row c_row
        col += 1
        
        # Term 4: a1 * b11 -> c1
        U[a_row * m + 1][col] = 1
        V[(b_row_off + 1) * p + 1][col] = 1
        W[1 * n + c_row][col] = 1
        col += 1
    
    # Now iterate over the blocks
    # A (3x4) split into A1 (cols 0,1) and A2 (cols 2,3)
    # B (4x6) split into B1 (rows 0,1) and B2 (rows 2,3)
    
    for a_block in range(2):  # 0 for A1, 1 for A2
        a_col_off = a_block * 2  # column offset in A
        b_row_off = a_block * 2  # row offset in B
        
        # For each 3x2x6 block, split B into three 2x2 subblocks
        for b_sub in range(3):  # 0, 1, 2 for columns 0-1, 2-3, 4-5
            b_col_off = b_sub * 2  # column offset in B subblock
            
            # For each 3x2x2 block (A_block 3x2 * B_subblock 2x2):
            # Top 2x2x2: rows 0,1 of A_block
            add_strassen_2x2x2(a_col_off, b_row_off, 0)
            
            # Bottom 1x2x2: row 2 of A_block
            add_schoolbook_1x2x2(2, b_row_off, 2)
    
    return {"U": U, "V": V, "W": W}
