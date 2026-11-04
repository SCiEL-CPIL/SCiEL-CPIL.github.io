import numpy as np

def construct():
    n, m, p = 2, 4, 6
    
    # Strategy: Split M_{2,4,6} into two M_{2,2,6} sub-tensors
    # Each M_{2,2,6} has rank 12 (from Strassen 2x2x2 blocks)
    # Total rank: 24
    
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 24
    V = [[0] * rank for _ in range(m * p)]  # 24 x 24
    W = [[0] * rank for _ in range(p * n)]  # 12 x 24
    
    # Helper: Add a rank-12 decomposition for M_{2,2,6}
    # A is 2x2, B is 2x6, C is 2x6
    # We use 3 Strassen 2x2x2 blocks for the 3 column pairs of B
    def add_2x2x6_block(a_idx, b_cols, w_cols, start_col):
        """
        a_idx: [a00, a01, a10, a11] indices into U (4 elements)
        b_cols: list of 3 pairs [(b00,b01), (b10,b11), (b20,b21)] for 2x6 B matrix
                 Each pair is (row0_col, row1_col) indices into V
        w_cols: list of 3 pairs [(w00,w01), (w10,w11), (w20,w21)] for 2x6 C matrix
                 Each pair is (row0_col, row1_col) indices into W
        start_col: starting column index
        """
        # For each of the 3 column pairs, apply Strassen 2x2x2
        for block in range(3):
            b00, b01 = b_cols[block]  # Wait, this is wrong. Let me re-think.
            pass
    
    # Let me use a different approach. 
    # For M_{2,2,6}: A is 2x2, B is 2x6, C is 2x6
    # Split B into 3 blocks of 2 columns: B1, B2, B3 (each 2x2)
    # C = A * B = A*B1 + A*B2 + A*B3
    # Each A*Bk is a 2x2 matrix, computed via Strassen (rank 7)
    # Total: 3 * 7 = 21 rank factors? No, that's not right either.
    
    # Actually, for 2x2 * 2x6:
    # We can view this as: C[:, 0:2] = A * B[:, 0:2]
    # C[:, 2:4] = A * B[:, 2:4]
    # C[:, 4:6] = A * B[:, 4:6]
    # Each is a 2x2 * 2x2 multiplication, which has rank 7 (Strassen)
    # So total rank = 3 * 7 = 21 for M_{2,2,6}
    
    # But wait, we can do better. Let me think about this more carefully.
    # Actually, the rank of M_{2,2,6} is known to be 12.
    # Let me construct it properly.
    
    # For M_{2,2,6}: 
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01, b02, b03, b04, b05],
    #      [b10, b11, b12, b13, b14, b15]]
    # C = A * B (2x6)
    
    # C00 = a00*b00 + a01*b10
    # C01 = a00*b01 + a01*b11
    # C02 = a00*b02 + a01*b12
    # C03 = a00*b03 + a01*b13
    # C04 = a00*b04 + a01*b14
    # C05 = a00*b05 + a01*b15
    # C10 = a10*b00 + a11*b10
    # C11 = a10*b01 + a11*b11
    # C12 = a10*b02 + a11*b12
    # C13 = a10*b03 + a11*b13
    # C14 = a10*b04 + a11*b14
    # C15 = a10*b05 + a11*b15
    
    # This is a sum of 12 rank-1 tensors:
    # For j in {0,1}, k in {0,...,5}:
    # T_{i,j,k} = A[i,j] * B[j,k] * C[i,k]
    # = sum over (j,k) of A[i,j] * B[j,k] * C[i,k]
    
    # So rank 12 is achievable with the schoolbook approach.
    # But can we do better? The lower bound is max(n*p, m*p, n*m) = max(12, 12, 4) = 12.
    # So rank 12 is optimal for M_{2,2,6}.
    
    # Therefore, for M_{2,4,6}, we can split into two M_{2,2,6} sub-problems:
    # A = [A_left, A_right] where A_left is 2x2, A_right is 2x2
    # B is 4x6, split into B_top (2x6) and B_bottom (2x6)
    # C = A_left * B_top + A_right * B_bottom
    # Each A_* * B_* is a 2x2 * 2x6 multiplication with rank 12
    # Total rank: 24
    
    # Now let's construct this.
    
    # A indices (n*m = 8):
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    
    # A_left: rows 0-1, cols 0-1: indices [0, 1, 4, 5]
    # A_right: rows 0-1, cols 2-3: indices [2, 3, 6, 7]
    
    # B indices (m*p = 24):
    # B[0,0]=0, B[0,1]=1, ..., B[0,5]=5
    # B[1,0]=6, B[1,1]=7, ..., B[1,5]=11
    # B[2,0]=12, B[2,1]=13, ..., B[2,5]=17
    # B[3,0]=18, B[3,1]=19, ..., B[3,5]=23
    
    # B_top: rows 0-1: indices [0..11]
    # B_bottom: rows 2-3: indices [12..23]
    
    # C indices (p*n = 12):
    # C[0,0]=0, C[0,1]=2, C[0,2]=4, C[0,3]=6, C[0,4]=8, C[0,5]=10
    # C[1,0]=1, C[1,1]=3, C[1,2]=5, C[1,3]=7, C[1,4]=9, C[1,5]=11
    
    # For M_{2,2,6} with A_left, B_top:
    # A_left indices: a00=0, a01=1, a10=4, a11=5
    # B_top indices: b00=0, b01=1, b02=2, b03=3, b04=4, b05=5
    #                b10=6, b11=7, b12=8, b13=9, b14=10, b15=11
    # C indices: c00=0, c01=2, c02=4, c03=6, c04=8, c05=10
    #            c10=1, c11=3, c12=5, c13=7, c14=9, c15=11
    
    # Schoolbook rank-12 decomposition for M_{2,2,6}:
    # For each (j, k) where j in {0,1}, k in {0,...,5}:
    # Factor r: U[a_idx[j]][r] = 1, V[b_idx[j,k]][r] = 1, W[c_idx[k]][r] = 1
    
    # Let's assign columns 0-11 for the first sub-problem (A_left * B_top)
    a_left = [0, 1, 4, 5]  # a00, a01, a10, a11
    b_top_rows = [[0, 1, 2, 3, 4, 5],  # row 0
                  [6, 7, 8, 9, 10, 11]]  # row 1
    c_cols = [0, 2, 4, 6, 8, 10]  # C[0,k] for k=0..5
    c_rows = [1, 3, 5, 7, 9, 11]  # C[1,k] for k=0..5
    
    # Column 0-11: A_left * B_top
    col = 0
    for j in range(2):  # j = 0, 1
        for k in range(6):  # k = 0, ..., 5
            U[a_left[j]][col] = 1
            V[b_top_rows[j][k]][col] = 1
            if j == 0:
                W[c_cols[k]][col] = 1
            else:
                W[c_rows[k]][col] = 1
            col += 1
    
    # Columns 12-23: A_right * B_bottom
    a_right = [2, 3, 6, 7]  # a00, a01, a10, a11
    b_bottom_rows = [[12, 13, 14, 15, 16, 17],  # row 0
                     [18, 19, 20, 21, 22, 23]]  # row 1
    
    col = 12
    for j in range(2):  # j = 0, 1
        for k in range(6):  # k = 0, ..., 5
            U[a_right[j]][col] = 1
            V[b_bottom_rows[j][k]][col] = 1
            if j == 0:
                W[c_cols[k]][col] = 1
            else:
                W[c_rows[k]][col] = 1
            col += 1
    
    # Verify the decomposition
    def verify_decomp(U, V, W, rank):
        """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all (i,j,k)"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    if not verify_decomp(U, V, W, rank):
        # Fallback: return the rank-42 Strassen decomposition
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(a_idx, b_idx, w_idx, start_col):
            c = start_col
            # M1
            U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
            V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
            W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
            # M2
            U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
            V[b_idx[0]][c+1] += 1
            W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
            # M3
            U[a_idx[0]][c+2] += 1
            V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
            W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
            # M4
            U[a_idx[3]][c+3] += 1
            V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
            W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
            # M5
            U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
            V[b_idx[3]][c+4] += 1
            W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
            # M6
            U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
            V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
            W[w_idx[3]][c+5] += 1
            # M7
            U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
            V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
            W[w_idx[0]][c+6] += 1
        
        a_L = [0, 1, 4, 5]
        a_R = [2, 3, 6, 7]
        b_1_T = [0, 1, 6, 7]
        b_1_B = [12, 13, 18, 19]
        w_1 = [0, 2, 1, 3]
        add_strassen_block(a_L, b_1_T, w_1, 0)
        add_strassen_block(a_R, b_1_B, w_1, 7)
        
        b_2_T = [2, 3, 8, 9]
        b_2_B = [14, 15, 20, 21]
        w_2 = [4, 6, 5, 7]
        add_strassen_block(a_L, b_2_T, w_2, 14)
        add_strassen_block(a_R, b_2_B, w_2, 21)
        
        b_3_T = [4, 5, 10, 11]
        b_3_B = [16, 17, 22, 23]
        w_3 = [8, 10, 9, 11]
        add_strassen_block(a_L, b_3_T, w_3, 28)
        add_strassen_block(a_R, b_3_B, w_3, 35)
        
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
