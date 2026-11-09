"""Matrix multiplication 6x6x6: Kronecker product of Strassen (2x2, rank 7) and schoolbook (3x3, rank 9) -> rank 63."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    # We decompose 6 = 2 * 3. The multiplication tensor T_{6,6,6} = T_{2,2,2} \otimes T_{3,3,3}.
    # Strassen gives rank 7 for T_{2,2,2}, schoolbook gives rank 9 for T_{3,3,3}.
    # Total rank = 7 * 9 = 63.
    
    # Strassen decomposition for 2x2 matrix multiplication:
    # U, V, W are 4x7 matrices (since 2*2=4 entries, 7 multiplications)
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # The 7 multiplications:
    # M1 = (A11) * (B11 + B22)
    # M2 = (A11 + A22) * B22
    # M3 = (A11 + A21) * B11
    # M4 = A22 * (B22 - B11)
    # M5 = (A21 + A22) * (B11 + B12)
    # M6 = (A11 - A21) * (B21 + B22)
    # M7 = (A12 - A22) * (B11 + B12)
    # C11 = M1 + M2 - M4 + M6
    # C12 = M3 + M5 - M6 - M7
    # C21 = M3 + M4 + M5 - M1
    # C22 = M1 + M2 - M5 + M7
    
    # U matrix: rows indexed by (i,j) where i in {0,1}, j in {0,1}; row = i*2+j
    # Columns are the 7 multiplications
    # U[i*2+j, c] gives coefficient of A[i,j] in multiplication c
    U_strassen = [
        [1, 1, 1, 0, 0, 1, 1],  # A[0,0] = A11
        [0, 0, 0, 1, 1, 0, -1], # A[0,1] = A12
        [0, 0, 1, 1, 1, -1, 0], # A[1,0] = A21
        [1, 1, 0, 1, 1, 0, -1], # A[1,1] = A22
    ]
    # Wait, let me re-derive carefully.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # M1 = A11 * (B11 + B22)
    # M2 = (A11 + A22) * B22
    # M3 = (A11 + A21) * B11
    # M4 = A22 * (B22 - B11)
    # M5 = (A21 + A22) * (B11 + B12)
    # M6 = (A11 - A21) * (B21 + B22)
    # M7 = (A12 - A22) * (B11 + B12)
    
    # U: coefficient of A[i,j] in factor c (c=0..6)
    # A[0,0]=A11: M1:1, M2:1, M3:1, M4:0, M5:0, M6:1, M7:0
    # A[0,1]=A12: M1:0, M2:0, M3:0, M4:0, M5:0, M6:0, M7:1
    # A[1,0]=A21: M1:0, M2:0, M3:1, M4:0, M5:1, M6:-1, M7:0
    # A[1,1]=A22: M1:0, M2:1, M3:0, M4:1, M5:1, M6:0, M7:-1
    U_strassen = [
        [1, 1, 1, 0, 0, 1, 0],  # A[0,0]
        [0, 0, 0, 0, 0, 0, 1],  # A[0,1]
        [0, 0, 1, 0, 1, -1, 0], # A[1,0]
        [0, 1, 0, 1, 1, 0, -1], # A[1,1]
    ]
    # V: coefficient of B[j,k] in factor c
    # B[0,0]=B11: M1:1, M2:0, M3:1, M4:-1, M5:1, M6:0, M7:1
    # B[0,1]=B12: M1:0, M2:0, M3:0, M4:0, M5:1, M6:0, M7:1
    # B[1,0]=B21: M1:0, M2:0, M3:0, M4:0, M5:0, M6:1, M7:0
    # B[1,1]=B22: M1:1, M2:1, M3:0, M4:1, M5:0, M6:1, M7:0
    V_strassen = [
        [1, 0, 1, -1, 1, 0, 1],  # B[0,0]
        [0, 0, 0, 0, 1, 0, 1],   # B[0,1]
        [0, 0, 0, 0, 0, 1, 0],   # B[1,0]
        [1, 1, 0, 1, 0, 1, 0],   # B[1,1]
    ]
    # W: coefficient in output C[i,k] from multiplication c
    # C[0,0]=C11 = M1 + M2 - M4 + M6
    # C[0,1]=C12 = M3 + M5 - M6 - M7
    # C[1,0]=C21 = M3 + M4 + M5 - M1
    # C[1,1]=C22 = M1 + M2 - M5 + M7
    # W[k*2+i, c] per trace convention: W[k*n+i, c] where n=2
    # So row index = k*2 + i
    # C[0,0]: i=0,k=0, row=0: M1:1, M2:1, M3:0, M4:-1, M5:0, M6:1, M7:0
    # C[0,1]: i=0,k=1, row=2: M1:0, M2:0, M3:1, M4:0, M5:1, M6:-1, M7:-1
    # C[1,0]: i=1,k=0, row=1: M1:-1, M2:0, M3:1, M4:1, M5:1, M6:0, M7:0
    # C[1,1]: i=1,k=1, row=3: M1:1, M2:1, M3:0, M4:0, M5:-1, M6:0, M7:1
    W_strassen = [
        [1, 1, 0, -1, 0, 1, 0],   # row 0: k=0,i=0 -> C[0,0]
        [-1, 0, 1, 1, 1, 0, 0],   # row 1: k=0,i=1 -> C[1,0]
        [0, 0, 1, 0, 1, -1, -1],  # row 2: k=1,i=0 -> C[0,1]
        [1, 1, 0, 0, -1, 0, 1],   # row 3: k=1,i=1 -> C[1,1]
    ]
    
    # Schoolbook for 3x3: rank 9
    U_sb = [[0]*9 for _ in range(9)]
    V_sb = [[0]*9 for _ in range(9)]
    W_sb = [[0]*9 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_sb[i*3+j][col] = 1
                V_sb[j*3+k][col] = 1
                W_sb[k*3+i][col] = 1
                col += 1
    
    # Now combine via Kronecker product.
    # The 6x6 multiplication: A is 6x6, B is 6x6, C = A*B is 6x6.
    # We view A as a 2x2 block matrix where each block is 3x3.
    # A = [[A00, A01], [A10, A11]] where each Aij is 3x3.
    # Similarly B = [[B00, B01], [B10, B11]].
    # C = A*B, each Cij is 3x3.
    
    # The Kronecker product decomposition:
    # For each Strassen multiplication c (0..6) and each schoolbook multiplication d (0..8):
    # The combined multiplication computes:
    # (sum over Strassen factors) * (sum over schoolbook factors)
    
    # U_6x6: shape (36, 63), row index = (block_i * 3 + local_i) * 3 + (block_j * 3 + local_j)
    # Actually, A[i,j] where i = bi*3 + li, j = bj*3 + lj
    # Row index in U = i*6 + j = (bi*3+li)*6 + (bj*3+lj)
    
    # For Strassen, the block index (bi, bj) corresponds to row bi*2+bj in U_strassen
    # For schoolbook, the local index (li, lj) corresponds to row li*3+lj in U_sb
    
    # U_6x6[i*6+j, c*9+d] = U_strassen[bi*2+bj, c] * U_sb[li*3+lj, d]
    
    rank_total = 7 * 9  # 63
    U = [[0] * rank_total for _ in range(36)]
    V = [[0] * rank_total for _ in range(36)]
    W = [[0] * rank_total for _ in range(36)]
    
    for c in range(7):
        for d in range(9):
            col_idx = c * 9 + d
            for bi in range(2):
                for bj in range(2):
                    u_str = U_strassen[bi * 2 + bj][c]
                    if u_str == 0:
                        continue
                    for li in range(3):
                        for lj in range(3):
                            u_sb_val = U_sb[li * 3 + lj][d]
                            if u_sb_val == 0:
                                continue
                            i = bi * 3 + li
                            j = bj * 3 + lj
                            U[i * 6 + j][col_idx] = u_str * u_sb_val
            
            for bj in range(2):
                for bk in range(2):
                    v_str = V_strassen[bj * 2 + bk][c]
                    if v_str == 0:
                        continue
                    for lj in range(3):
                        for lk in range(3):
                            v_sb_val = V_sb[lj * 3 + lk][d]
                            if v_sb_val == 0:
                                continue
                            j = bj * 3 + lj
                            k = bk * 3 + lk
                            V[j * 6 + k][col_idx] = v_str * v_sb_val
            
            for bk in range(2):
                for bi in range(2):
                    w_str = W_strassen[bk * 2 + bi][c]
                    if w_str == 0:
                        continue
                    for lk in range(3):
                        for li in range(3):
                            w_sb_val = W_sb[lk * 3 + li][d]
                            if w_sb_val == 0:
                                continue
                            k = bk * 3 + lk
                            i = bi * 3 + li
                            W[k * 6 + i][col_idx] = w_str * w_sb_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
