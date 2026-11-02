"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is 56. We target rank 50.
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen decomposition
    # a_idx: indices in U for the 2x2 matrix A (row-major: 0,1,4,5 relative to global, but we pass absolute)
    # a_idx = [u_00, u_01, u_10, u_11]
    # b_idx: indices in V for the 2x2 matrix B
    # b_idx = [v_00, v_01, v_10, v_11]
    # w_idx: indices in W for the 2x2 output C
    # w_idx = [w_00, w_01, w_10, w_11]
    def add_strassen(a_idx, b_idx, w_idx, start_col):
        # M1 = (A00 + A11) * (B00 + B11)
        U[a_idx[0]][start_col] = 1
        U[a_idx[2]][start_col] = 1
        V[b_idx[0]][start_col] = 1
        V[b_idx[2]][start_col] = 1
        W[w_idx[0]][start_col] = 1
        W[w_idx[3]][start_col] = 1
        W[w_idx[1]][start_col] = -1
        W[w_idx[2]][start_col] = 1
        
        # M2 = (A10 + A11) * B00
        U[a_idx[2]][start_col+1] = 1
        U[a_idx[3]][start_col+1] = 1
        V[b_idx[0]][start_col+1] = 1
        W[w_idx[2]][start_col+1] = 1
        W[w_idx[3]][start_col+1] = -1
        
        # M3 = A00 * (B10 - B11)
        U[a_idx[0]][start_col+2] = 1
        V[b_idx[2]][start_col+2] = 1
        V[b_idx[3]][start_col+2] = -1
        W[w_idx[1]][start_col+2] = 1
        W[w_idx[3]][start_col+2] = 1
        
        # M4 = A11 * (B01 - B00)
        U[a_idx[3]][start_col+3] = 1
        V[b_idx[1]][start_col+3] = 1
        V[b_idx[0]][start_col+3] = -1
        W[w_idx[0]][start_col+3] = 1
        W[w_idx[2]][start_col+3] = 1
        
        # M5 = (A00 + A01) * B11
        U[a_idx[0]][start_col+4] = 1
        U[a_idx[1]][start_col+4] = 1
        V[b_idx[3]][start_col+4] = 1
        W[w_idx[0]][start_col+4] = -1
        W[w_idx[1]][start_col+4] = 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[a_idx[2]][start_col+5] = 1
        U[a_idx[0]][start_col+5] = -1
        V[b_idx[0]][start_col+5] = 1
        V[b_idx[1]][start_col+5] = 1
        W[w_idx[3]][start_col+5] = 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[a_idx[1]][start_col+6] = 1
        U[a_idx[3]][start_col+6] = -1
        V[b_idx[2]][start_col+6] = 1
        V[b_idx[3]][start_col+6] = 1
        W[w_idx[0]][start_col+6] = 1
        W[w_idx[1]][start_col+6] = 1
        
        return start_col + 7

    # Helper to add a 2x2x1 decomposition
    # A is 2x2, B is 2x1 (column vector), C is 2x1
    def add_221(a_idx, b_idx, w_idx, start_col):
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        U[a_idx[0]][start_col] = 1
        V[b_idx[0]][start_col] = 1
        W[w_idx[0]][start_col] = 1
        
        U[a_idx[1]][start_col+1] = 1
        V[b_idx[1]][start_col+1] = 1
        W[w_idx[0]][start_col+1] = 1
        
        U[a_idx[2]][start_col+2] = 1
        V[b_idx[0]][start_col+2] = 1
        W[w_idx[1]][start_col+2] = 1
        
        U[a_idx[3]][start_col+3] = 1
        V[b_idx[1]][start_col+3] = 1
        W[w_idx[1]][start_col+3] = 1
        
        return start_col + 4

    # Indices for A_L (rows 0,1 of A, cols 0,1)
    # U indices: i*m+j
    # A_L[0,0] -> U[0]
    # A_L[0,1] -> U[1]
    # A_L[1,0] -> U[4]
    # A_L[1,1] -> U[5]
    aL_idx = [0, 1, 4, 5]
    
    # Indices for A_R (rows 0,1 of A, cols 2,3)
    # A_R[0,0] (A[0,2]) -> U[2]
    # A_R[0,1] (A[0,3]) -> U[3]
    # A_R[1,0] (A[1,2]) -> U[6]
    # A_R[1,1] (A[1,3]) -> U[7]
    aR_idx = [2, 3, 6, 7]
    
    # Indices for B_T (rows 0,1 of B)
    # V indices: j*p+k
    # B_T[0,0] -> V[0]
    # B_T[0,1] -> V[1]
    # B_T[0,2] -> V[2]
    # B_T[0,3] -> V[3]
    # B_T[0,4] -> V[4]
    # B_T[0,5] -> V[5]
    # B_T[0,6] -> V[6]
    
    # B_T[1,0] -> V[7]
    # B_T[1,1] -> V[8]
    # B_T[1,2] -> V[9]
    # B_T[1,3] -> V[10]
    # B_T[1,4] -> V[11]
    # B_T[1,5] -> V[12]
    # B_T[1,6] -> V[13]
    
    # Indices for B_B (rows 2,3 of B)
    # B_B[0,0] (B[2,0]) -> V[14]
    # B_B[0,1] (B[2,1]) -> V[15]
    # ...
    # B_B[0,k] -> V[14+k]
    # B_B[1,k] (B[3,k]) -> V[21+k]
    
    # Output W indices: k*n+i
    # W[0,0] -> W[0]
    # W[0,1] -> W[1]
    # W[1,0] -> W[2]
    # W[1,1] -> W[3]
    # ...
    # W[k, i] -> W[k*2+i]
    
    # 1. A_L * B_T[0:2, 0:4] (2x2 by 2x4)
    # Split into two 2x2x2
    # B_T_left1: cols 0,1
    bT1_idx = [0, 1, 7, 8] # B[0,0], B[0,1], B[1,0], B[1,1]
    w_out1_idx = [0, 1, 2, 3] # C[0,0], C[0,1], C[1,0], C[1,1]
    col = add_strassen(aL_idx, bT1_idx, w_out1_idx, col)
    
    # B_T_left2: cols 2,3
    bT2_idx = [2, 3, 9, 10] # B[0,2], B[0,3], B[1,2], B[1,3]
    w_out2_idx = [4, 5, 6, 7] # C[0,2], C[0,3], C[1,2], C[1,3]
    col = add_strassen(aL_idx, bT2_idx, w_out2_idx, col)
    
    # 2. A_L * B_T[0:2, 4:7] (2x2 by 2x3)
    # Split into 2x2x2 (cols 4,5) and 2x2x1 (col 6)
    # B_T_mid: cols 4,5
    bT3_idx = [4, 5, 11, 12]
    w_out3_idx = [8, 9, 10, 11] # C[0,4], C[0,5], C[1,4], C[1,5]
    col = add_strassen(aL_idx, bT3_idx, w_out3_idx, col)
    
    # B_T_tail: col 6
    bT4_idx = [6, 13] # B[0,6], B[1,6]
    w_out4_idx = [12, 13] # C[0,6], C[1,6]
    col = add_221(aL_idx, bT4_idx, w_out4_idx, col)
    
    # 3. A_R * B_B[2:4, 0:4] (2x2 by 2x4)
    # B_B_left1: cols 0,1
    bB1_idx = [14, 15, 21, 22] # B[2,0], B[2,1], B[3,0], B[3,1]
    col = add_strassen(aR_idx, bB1_idx, w_out1_idx, col)
    
    # B_B_left2: cols 2,3
    bB2_idx = [16, 17, 23, 24] # B[2,2], B[2,3], B[3,2], B[3,3]
    col = add_strassen(aR_idx, bB2_idx, w_out2_idx, col)
    
    # 4. A_R * B_B[2:4, 4:7] (2x2 by 2x3)
    # B_B_mid: cols 4,5
    bB3_idx = [18, 19, 25, 26]
    col = add_strassen(aR_idx, bB3_idx, w_out3_idx, col)
    
    # B_B_tail: col 6
    bB4_idx = [20, 27]
    col = add_221(aR_idx, bB4_idx, w_out4_idx, col)
    
    # col should be 50
    # 7*6 + 4*2 = 42 + 8 = 50. Correct.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
