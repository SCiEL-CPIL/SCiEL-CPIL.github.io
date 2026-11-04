import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Decompose into blocks
    # A: 4x5 -> rows 2,2; cols 2,2,1
    # B: 5x6 -> rows 2,2,1; cols 2,2,2
    
    # We will construct U, V, W matrices
    # Rank will be sum of ranks of blocks
    
    rank_total = 0
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a block decomposition
    def add_block(n_b, m_b, p_b, offset_i, offset_j, offset_k, rank_b, U_b, V_b, W_b):
        # U_b is n_b x rank_b
        # V_b is m_b x rank_b
        # W_b is p_b x rank_b
        pass

    # We need to map the global indices to local block indices
    # Global U: (n*m, rank)
    # Global V: (m*p, rank)
    # Global W: (p*n, rank)
    
    # Let's just build the full matrices directly
    # Max rank estimate: 12 * 7 + 24 = 108
    # Let's allocate 108 columns
    
    R = 108
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    col = 0
    
    # Strassen 2x2x2 decomposition
    # U_s, V_s, W_s for 2x2x2
    U_s = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]
    ]) # This is identity, not Strassen
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U_s (4x7), V_s (4x7), W_s (4x7)
    # Indices: 0,1 for row; 0,1 for col in U/V/W?
    # U maps A indices (i, j) -> i is 0..1, j is 0..1.
    # Actually U is (n*m, R). For 2x2, n=2, m=2, so 4 rows.
    # Row index in U: i*2 + j.
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1: A11+A22, B11+B22
    U_s[0, 0] = 1 # A11 (i=0, j=0)
    U_s[3, 0] = 1 # A22 (i=1, j=1)
    V_s[0, 0] = 1 # B11 (j=0, k=0)
    V_s[3, 0] = 1 # B22 (j=1, k=1)
    W_s[0, 0] = 1 # C11 (k=0, i=0)
    W_s[3, 0] = 1 # C22 (k=1, i=1)
    W_s[1, 0] = -1 # C12 (k=1, i=0)
    W_s[2, 0] = -1 # C21 (k=0, i=1)
    
    # M2: A21+A22, B11
    U_s[2, 1] = 1 # A21 (i=1, j=0)
    U_s[3, 1] = 1 # A22 (i=1, j=1)
    V_s[0, 1] = 1 # B11
    W_s[2, 1] = 1 # C21
    W_s[3, 1] = 1 # C22
    
    # M3: A11, B21-B22
    U_s[0, 2] = 1 # A11
    V_s[2, 2] = 1 # B21 (j=1, k=0)
    V_s[3, 2] = -1 # B22 (j=1, k=1)
    W_s[0, 2] = 1 # C11
    W_s[1, 2] = 1 # C12
    
    # M4: A22, B12-B11
    U_s[3, 3] = 1 # A22
    V_s[1, 3] = 1 # B12 (j=0, k=1)
    V_s[0, 3] = -1 # B11
    W_s[2, 3] = 1 # C21
    W_s[3, 3] = 1 # C22
    
    # M5: A11+A12, B22
    U_s[0, 4] = 1 # A11
    U_s[1, 4] = 1 # A12 (i=0, j=1)
    V_s[3, 4] = 1 # B22
    W_s[1, 4] = 1 # C12
    W_s[3, 4] = 1 # C22
    
    # M6: A21-A11, B11+B12
    U_s[2, 5] = 1 # A21
    U_s[0, 5] = -1 # A11
    V_s[0, 5] = 1 # B11
    V_s[1, 5] = 1 # B12
    W_s[2, 5] = 1 # C21
    W_s[3, 5] = 1 # C22
    
    # M7: A12-A22, B21+B22
    U_s[1, 6] = 1 # A12
    U_s[3, 6] = -1 # A22
    V_s[2, 6] = 1 # B21
    V_s[3, 6] = 1 # B22
    W_s[0, 6] = 1 # C11
    W_s[1, 6] = 1 # C12

    # Now map blocks
    # A blocks:
    # A11 (2x2): i=0,1 j=0,1
    # A12 (2x2): i=0,1 j=2,3
    # A13 (2x1): i=0,1 j=4
    # A21 (2x2): i=2,3 j=0,1
    # A22 (2x2): i=2,3 j=2,3
    # A23 (2x1): i=2,3 j=4
    
    # B blocks:
    # B11 (2x2): j=0,1 k=0,1
    # B12 (2x2): j=0,1 k=2,3
    # B13 (2x2): j=0,1 k=4,5
    # B21 (2x2): j=2,3 k=0,1
    # B22 (2x2): j=2,3 k=2,3
    # B23 (2x2): j=2,3 k=4,5
    # B31 (1x2): j=4 k=0,1
    # B32 (1x2): j=4 k=2,3
    # B33 (1x2): j=4 k=4,5
    
    # Product C blocks (4x6):
    # C11 (2x2): i=0,1 k=0,1
    # C12 (2x2): i=0,1 k=2,3
    # C13 (2x2): i=0,1 k=4,5
    # C21 (2x2): i=2,3 k=0,1
    # C22 (2x2): i=2,3 k=2,3
    # C23 (2x2): i=2,3 k=4,5
    
    # Terms:
    # A11*B11 -> C11 (2x2x2)
    # A12*B12 -> C12 (2x2x2)
    # A13*B13 -> C13 (2x1x2)
    # A21*B21 -> C21 (2x2x2)
    # A22*B22 -> C22 (2x2x2)
    # A23*B23 -> C23 (2x1x2)
    # A11*B21 -> C11? No, A11 is i=0,1 j=0,1. B21 is j=2,3. Mismatch.
    
    # The sum is C = A B.
    # C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # j ranges 0..4.
    # j=0,1 (B11, B21, B31)
    # j=2,3 (B12, B22, B32)
    # j=4 (B13, B23, B33)
    
    # So:
    # C11 (i=0,1 k=0,1) gets:
    # A11 (j=0,1) * B11 (j=0,1) -> 2x2x2
    # A12 (j=2,3) * B21 (j=2,3) -> 2x2x2
    # A13 (j=4) * B31 (j=4) -> 2x1x2
    
    # C12 (i=0,1 k=2,3) gets:
    # A11 * B12 -> 2x2x2
    # A12 * B22 -> 2x2x2
    # A13 * B32 -> 2x1x2
    
    # C13 (i=0,1 k=4,5) gets:
    # A11 * B13 -> 2x2x2
    # A12 * B23 -> 2x2x2
    # A13 * B33 -> 2x1x2
    
    # Similarly for C21, C22, C23 with A21, A22, A23.
    
    # Total 2x2x2 blocks: 3 (cols) * 3 (rows of A) * 2 (rows of C)?
    # Rows of A: 2 blocks (top, bot).
    # Cols of C: 3 blocks (L, M, R).
    # For each (A_row_block, C_col_block), we have 3 terms (j=0,1; j=2,3; j=4).
    # j=0,1 and j=2,3 are 2x2x2. j=4 is 2x1x2.
    # So 2 (A rows) * 3 (C cols) * 2 (2x2x2 types) = 12 blocks of 2x2x2.
    # 2 (A rows) * 3 (C cols) * 1 (2x1x2 type) = 6 blocks of 2x1x2.
    
    # Rank of 2x2x2 is 7.
    # Rank of 2x1x2 is 4 (schoolbook 2*1*2=4).
    # Total rank = 12 * 7 + 6 * 4 = 84 + 24 = 108.
    
    # Implement mapping
    def map_2x2x2(offset_i, offset_j, offset_k, col_start):
        # offset_i: 0 or 2
        # offset_j: 0, 2, or 4 (but 4 is 2x1x2)
        # offset_k: 0, 2, or 4
        
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    if U_s[i*2+j, r] != 0:
                        U[(offset_i + i)*2 + (offset_j + j), col_start + r] = U_s[i*2+j, r]
            for j in range(2):
                for k in range(2):
                    if V_s[j*2+k, r] != 0:
                        V[(offset_j + j)*2 + (offset_k + k), col_start + r] = V_s[j*2+k, r]
            for k in range(2):
                for i in range(2):
                    if W_s[k*2+i, r] != 0:
                        W[(offset_k + k)*2 + (offset_i + i), col_start + r] = W_s[k*2+i, r]

    def map_2x1x2(offset_i, offset_j, offset_k, col_start):
        # offset_i: 0 or 2
        # offset_j: 4
        # offset_k: 0, 2, or 4
        # Rank 4
        # U: 2x4, V: 1x4, W: 2x4
        # U_s_212[i, r] = 1 if i==r? No.
        # T_{i, j, k} = 1.
        # i in 0,1. j=0. k in 0,1.
        # R=4.
        # r=0: i=0, k=0
        # r=1: i=0, k=1
        # r=2: i=1, k=0
        # r=3: i=1, k=1
        for r in range(4):
            i = r // 2
            k = r % 2
            U[(offset_i + i)*2 + (offset_j), col_start + r] = 1
            V[(offset_j)*2 + (offset_k), col_start + r] = 1
            W[(offset_k + k)*2 + (offset_i + i), col_start + r] = 1

    # Iterate
    # A_row_block 0 (i=0,1)
    # C_col_block 0 (k=0,1)
    # j=0,1 (offset_j=0)
    map_2x2x2(0, 0, 0, col); col += 7
    # j=2,3 (offset_j=2)
    map_2x2x2(0, 2, 0, col); col += 7
    # j=4 (offset_j=4)
    map_2x1x2(0, 4, 0, col); col += 4
    
    # C_col_block 1 (k=2,3)
    map_2x2x2(0, 0, 2, col); col += 7
    map_2x2x2(0, 2, 2, col); col += 7
    map_2x1x2(0, 4, 2, col); col += 4
    
    # C_col_block 2 (k=4,5)
    map_2x2x2(0, 0, 4, col); col += 7
    map_2x2x2(0, 2, 4, col); col += 7
    map_2x1x2(0, 4, 4, col); col += 4
    
    # A_row_block 1 (i=2,3)
    # C_col_block 0 (k=0,1)
    map_2x2x2(2, 0, 0, col); col += 7
    map_2x2x2(2, 2, 0, col); col += 7
    map_2x1x2(2, 4, 0, col); col += 4
    
    # C_col_block 1 (k=2,3)
    map_2x2x2(2, 0, 2, col); col += 7
    map_2x2x2(2, 2, 2, col); col += 7
    map_2x1x2(2, 4, 2, col); col += 4
    
    # C_col_block 2 (k=4,5)
    map_2x2x2(2, 0, 4, col); col += 7
    map_2x2x2(2, 2, 4, col); col += 7
    map_2x1x2(2, 4, 4, col); col += 4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
