import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Total rank estimate: 48
    # We will build it block by block.
    
    # Initialize U, V, W as lists of lists of zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We don't know the exact rank yet, so we'll accumulate columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block(U_block, V_block, W_block, u_offset, v_offset, w_offset):
        # u_offset, v_offset, w_offset are (row_start, col_start) for the block in the full matrix
        # U_block is (block_rows, block_rank)
        # We need to add to the global U, V, W
        pass

    # Helper to get Strassen U, V, W for 2x2
    def get_strassen_2x2():
        # Returns U, V, W for 2x2 multiplication
        # U, V, W are 4x7
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1
        
        return U, V, W

    U_str, V_str, W_str = get_strassen_2x2()
    
    # Helper to add a 2x2 Strassen block to the global matrices
    # a_row_start, a_col_start: top-left of A block
    # b_row_start, b_col_start: top-left of B block
    # c_row_start, c_col_start: top-left of C block
    def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c):
        # U indices for A block: (a_r + i, a_c + j)
        # V indices for B block: (b_r + i, b_c + j)
        # W indices for C block: (c_r + i, c_c + j)
        
        # We need to extend U_cols, V_cols, W_cols by 7 columns
        # Or we can just append to lists
        
        for col in range(7):
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            # Map U_str
            for i in range(2):
                for j in range(2):
                    val = U_str[i*2+j][col]
                    if val != 0:
                        u_idx = (a_r + i) * m + (a_c + j)
                        u_col[u_idx] = val
                        
            # Map V_str
            for i in range(2):
                for j in range(2):
                    val = V_str[i*2+j][col]
                    if val != 0:
                        v_idx = (b_r + i) * p + (b_c + j)
                        v_col[v_idx] = val
                        
            # Map W_str
            for i in range(2):
                for j in range(2):
                    val = W_str[i*2+j][col]
                    if val != 0:
                        w_idx = (c_r + i) * n + (c_c + j)
                        w_col[w_idx] = val
                        
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

    # Helper to add a 1x4 or 1x2 schoolbook block
    # A is 1xM, B is MxP
    def add_schoolbook_block(a_r, a_c, b_r, b_c, c_r, c_c, m_size, p_size):
        # A is 1 x m_size
        # B is m_size x p_size
        # C is 1 x p_size
        # Rank is p_size? No, rank is m_size * p_size?
        # C_j = sum_k A_k B_kj
        # This is p_size dot products of length m_size.
        # Each dot product is a rank 1 tensor?
        # T_{i,j,k} = A_i B_jk.
        # If i is fixed, T_{j,k} = sum_l A_l B_lk.
        # This is a bilinear map.
        # We can decompose it into p_size terms?
        # Term j: sum_l A_l B_lj.
        # U: A_l. V: B_lj. W: C_j.
        # For each l, we have a term?
        # No, the sum is over l.
        # So for each j, we need to sum over l.
        # This requires m_size * p_size terms if we do it naively?
        # Wait, rank of M_{1, m, p} is m?
        # Let's check M_{1, 2, 2}.
        # C_0 = A_0 B_00 + A_1 B_10
        # C_1 = A_0 B_01 + A_1 B_11
        # Rank 2?
        # Term 0: A_0 * (B_00, B_01) -> C_0 += A_0 B_00, C_1 += A_0 B_01
        # Term 1: A_1 * (B_10, B_11) -> C_0 += A_1 B_10, C_1 += A_1 B_11
        # Yes, rank is m_size.
        
        for l in range(m_size):
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            # U: A_l
            u_idx = (a_r) * m + (a_c + l)
            u_col[u_idx] = 1
            
            # V: B_l, * (all p columns)
            for j in range(p_size):
                v_idx = (b_r + l) * p + (b_c + j)
                v_col[v_idx] = 1
                
            # W: C, * (all p columns)
            for j in range(p_size):
                w_idx = (c_r) * n + (c_c + j)
                w_col[w_idx] = 1
                
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

    # 1. C_top_left (2x4x4)
    # Split into two 2x2x2 Strassen blocks?
    # C_TL = A_TL B_TL + A_TR B_BL
    # This is two 2x2 multiplications.
    # Block 1: A_TL (2x2) * B_TL (2x2) -> C_TL_part1 (2x2)
    # Block 2: A_TR (2x2) * B_BL (2x2) -> C_TL_part2 (2x2)
    # We add them to the same C block.
    
    # A_TL: a_r=0, a_c=0
    # B_TL: b_r=0, b_c=0
    # C_TL: c_r=0, c_c=0
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # A_TR: a_r=0, a_c=2
    # B_BL: b_r=2, b_c=0
    # C_TL: c_r=0, c_c=0
    add_strassen_block(0, 2, 2, 0, 0, 0)
    
    # C_TR = A_TL B_TR + A_TR B_BR
    # A_TL: a_r=0, a_c=0
    # B_TR: b_r=0, b_c=2
    # C_TR: c_r=0, c_c=2
    add_strassen_block(0, 0, 0, 2, 0, 2)
    
    # A_TR: a_r=0, a_c=2
    # B_BR: b_r=2, b_c=2
    # C_TR: c_r=0, c_c=2
    add_strassen_block(0, 2, 2, 2, 0, 2)
    
    # 2. C_top_right (2x4x2)
    # C = A_TL B_TL_right + A_TR B_BL_right
    # B_right is cols 4,5.
    # B_TL_right: b_r=0, b_c=4
    # B_BL_right: b_r=2, b_c=4
    
    # A_TL: a_r=0, a_c=0
    # C_TR_right: c_r=0, c_c=4
    add_strassen_block(0, 0, 0, 4, 0, 4)
    
    # A_TR: a_r=0, a_c=2
    # C_TR_right: c_r=0, c_c=4
    add_strassen_block(0, 2, 2, 4, 0, 4)
    
    # 3. C_bot_left (1x4x4)
    # A_bot: a_r=2, a_c=0
    # B_left: b_r=0, b_c=0
    # C_bot_left: c_r=2, c_c=0
    add_schoolbook_block(2, 0, 0, 0, 2, 0, 4, 4)
    
    # 4. C_bot_right (1x4x2)
    # A_bot: a_r=2, a_c=0
    # B_right: b_r=0, b_c=4
    # C_bot_right: c_r=2, c_c=4
    add_schoolbook_block(2, 0, 0, 4, 2, 4, 4, 2)
    
    # Convert lists of columns to matrices
    rank = len(U_cols)
    U = [col for col in U_cols]
    # Wait, U_cols are columns. We need rows.
    # U is (n*m, rank).
    U_mat = [[0] * rank for _ in range(n * m)]
    for r in range(n * m):
        for c in range(rank):
            U_mat[r][c] = U_cols[c][r]
            
    V_mat = [[0] * rank for _ in range(m * p)]
    for r in range(m * p):
        for c in range(rank):
            V_mat[r][c] = V_cols[c][r]
            
    W_mat = [[0] * rank for _ in range(p * n)]
    for r in range(p * n):
        for c in range(rank):
            W_mat[r][c] = W_cols[c][r]
            
    return {"U": U_mat, "V": V_mat, "W": W_mat}
