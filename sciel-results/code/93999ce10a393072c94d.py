import time
import sys

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    # U: (n*m, rank) = (6, 28)
    # V: (m*p, rank) = (15, 28)
    # W: (p*n, rank) = (10, 28)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to set a rank-1 term
    # u: index in U (0..5)
    # v: index in V (0..14)
    # w: index in W (0..9)
    # coeff: 1.0
    # col: column index
    
    def add_term(col, u_idx, v_idx, w_idx):
        U[u_idx][col] = 1.0
        V[v_idx][col] = 1.0
        W[w_idx][col] = 1.0

    def add_strassen_block(col_start, b_row1_col1, b_row1_col2, b_row2_col1, b_row2_col2, b_row3_col1, b_row3_col2):
        # A indices:
        # a11=0, a12=1, a13=2
        # a21=3, a22=4, a23=5
        
        # B indices (m*p):
        # b11=0, b12=1, b13=2, b14=3, b15=4
        # b21=5, b22=6, b23=7, b24=8, b25=9
        # b31=10, b32=11, b33=12, b34=13, b35=14
        
        # Strassen M1..M7
        # M1 = a11 * (b11 + b22)
        U[0][col_start] = 1.0
        V[b_row1_col1][col_start] = 1.0
        V[b_row2_col2][col_start] = 1.0
        # W contributions for M1:
        # z11 += M1, z22 += M1
        W[0][col_start] = 1.0 # k*n+i -> 0*2+0=0 (z11)
        W[3][col_start] = 1.0 # k*n+i -> 1*2+1=3 (z22)
        
        # M2 = (a11 + a21) * b22
        U[0][col_start+1] = 1.0
        U[3][col_start+1] = 1.0
        V[b_row2_col2][col_start+1] = 1.0
        # z21 += M2, z22 -= M2
        W[2][col_start+1] = 1.0 # 1*2+0=2 (z21)
        W[3][col_start+1] = -1.0
        
        # M3 = (a12 + a22) * b11
        U[1][col_start+2] = 1.0
        U[4][col_start+2] = 1.0
        V[b_row1_col1][col_start+2] = 1.0
        # z12 += M3, z22 += M3
        W[1][col_start+2] = 1.0 # 0*2+1=1 (z12)
        W[3][col_start+2] = 1.0
        
        # M4 = a22 * (b12 - b21)
        U[4][col_start+3] = 1.0
        V[b_row1_col2][col_start+3] = 1.0
        V[b_row2_col1][col_start+3] = -1.0
        # z11 += M4, z21 += M4
        W[0][col_start+3] = 1.0
        W[2][col_start+3] = 1.0
        
        # M5 = (a11 + a12) * b21
        U[0][col_start+4] = 1.0
        U[1][col_start+4] = 1.0
        V[b_row2_col1][col_start+4] = 1.0
        # z11 -= M5, z12 += M5
        W[0][col_start+4] = -1.0
        W[1][col_start+4] = 1.0
        
        # M6 = (a21 - a11) * (b11 + b12)
        U[3][col_start+5] = 1.0
        U[0][col_start+5] = -1.0
        V[b_row1_col1][col_start+5] = 1.0
        V[b_row1_col2][col_start+5] = 1.0
        # z22 += M6
        W[3][col_start+5] = 1.0
        
        # M7 = (a12 - a22) * (b21 + b22)
        U[1][col_start+6] = 1.0
        U[4][col_start+6] = -1.0
        V[b_row2_col1][col_start+6] = 1.0
        V[b_row2_col2][col_start+6] = 1.0
        # z11 += M7
        W[0][col_start+6] = 1.0
        
        # Outer products O1..O4
        # O1 = a13 * b31
        U[2][col_start+7] = 1.0
        V[b_row3_col1][col_start+7] = 1.0
        W[0][col_start+7] = 1.0 # z11
        
        # O2 = a13 * b32
        U[2][col_start+8] = 1.0
        V[b_row3_col2][col_start+8] = 1.0
        W[1][col_start+8] = 1.0 # z12
        
        # O3 = a23 * b31
        U[5][col_start+9] = 1.0
        V[b_row3_col1][col_start+9] = 1.0
        W[2][col_start+9] = 1.0 # z21
        
        # O4 = a23 * b32
        U[5][col_start+10] = 1.0
        V[b_row3_col2][col_start+10] = 1.0
        W[3][col_start+10] = 1.0 # z22

    # Block 1: Cols 0,1 of C (indices 0,1 in W for row 0, 2,3 for row 1)
    # B cols 0,1. Rows 0,1 are 0,1; 2,3. Row 2 is 4,5.
    # b11=0, b12=1, b21=5, b22=6, b31=10, b32=11
    add_strassen_block(0, 0, 1, 5, 6, 10, 11)
    
    # Block 2: Cols 2,3 of C (indices 4,5 in W for row 0, 6,7 for row 1)
    # Wait, W index is k*n+i.
    # C is 2x5.
    # C[0,0]=W[0], C[0,1]=W[1], C[0,2]=W[2], C[0,3]=W[3], C[0,4]=W[4]
    # C[1,0]=W[5], C[1,1]=W[6], C[1,2]=W[7], C[1,3]=W[8], C[1,4]=W[9]
    # My previous W mapping in add_strassen_block was:
    # W[0] -> z11 (C[0,0])
    # W[1] -> z12 (C[0,1])
    # W[2] -> z21 (C[1,0])
    # W[3] -> z22 (C[1,1])
    # This assumes the block outputs to C[0,0..1] and C[1,0..1].
    # For Block 2, we need to output to C[0,2..3] and C[1,2..3].
    # So W indices should be 2,3 for row 0 and 7,8 for row 1.
    
    # I need to generalize the W indices in add_strassen_block.
    
    def add_strassen_block_general(col_start, w_11, w_12, w_21, w_22, 
                                   b_row1_col1, b_row1_col2, b_row2_col1, b_row2_col2, 
                                   b_row3_col1, b_row3_col2):
        # M1
        U[0][col_start] = 1.0
        V[b_row1_col1][col_start] = 1.0
        V[b_row2_col2][col_start] = 1.0
        W[w_11][col_start] = 1.0
        W[w_22][col_start] = 1.0
        
        # M2
        U[0][col_start+1] = 1.0
        U[3][col_start+1] = 1.0
        V[b_row2_col2][col_start+1] = 1.0
        W[w_21][col_start+1] = 1.0
        W[w_22][col_start+1] = -1.0
        
        # M3
        U[1][col_start+2] = 1.0
        U[4][col_start+2] = 1.0
        V[b_row1_col1][col_start+2] = 1.0
        W[w_12][col_start+2] = 1.0
        W[w_22][col_start+2] = 1.0
        
        # M4
        U[4][col_start+3] = 1.0
        V[b_row1_col2][col_start+3] = 1.0
        V[b_row2_col1][col_start+3] = -1.0
        W[w_11][col_start+3] = 1.0
        W[w_21][col_start+3] = 1.0
        
        # M5
        U[0][col_start+4] = 1.0
        U[1][col_start+4] = 1.0
        V[b_row2_col1][col_start+4] = 1.0
        W[w_11][col_start+4] = -1.0
        W[w_12][col_start+4] = 1.0
        
        # M6
        U[3][col_start+5] = 1.0
        U[0][col_start+5] = -1.0
        V[b_row1_col1][col_start+5] = 1.0
        V[b_row1_col2][col_start+5] = 1.0
        W[w_22][col_start+5] = 1.0
        
        # M7
        U[1][col_start+6] = 1.0
        U[4][col_start+6] = -1.0
        V[b_row2_col1][col_start+6] = 1.0
        V[b_row2_col2][col_start+6] = 1.0
        W[w_11][col_start+6] = 1.0
        
        # O1
        U[2][col_start+7] = 1.0
        V[b_row3_col1][col_start+7] = 1.0
        W[w_11][col_start+7] = 1.0
        
        # O2
        U[2][col_start+8] = 1.0
        V[b_row3_col2][col_start+8] = 1.0
        W[w_12][col_start+8] = 1.0
        
        # O3
        U[5][col_start+9] = 1.0
        V[b_row3_col1][col_start+9] = 1.0
        W[w_21][col_start+9] = 1.0
        
        # O4
        U[5][col_start+10] = 1.0
        V[b_row3_col2][col_start+10] = 1.0
        W[w_22][col_start+10] = 1.0

    # Block 1: Cols 0,1
    # W: C[0,0]=0, C[0,1]=1, C[1,0]=5, C[1,1]=6
    add_strassen_block_general(0, 0, 1, 5, 6, 0, 1, 5, 6, 10, 11)
    
    # Block 2: Cols 2,3
    # W: C[0,2]=2, C[0,3]=3, C[1,2]=7, C[1,3]=8
    # B: Cols 2,3.
    # b13=2, b14=3, b23=7, b24=8, b33=12, b34=13
    add_strassen_block_general(11, 2, 3, 7, 8, 2, 3, 7, 8, 12, 13)
    
    # Block 3: Col 4 (Rank 6)
    # W: C[0,4]=4, C[1,4]=9
    # B: Col 4.
    # b15=4, b25=9, b35=14
    # Schoolbook S1..S4
    # S1 = a11 * b15
    U[0][22] = 1.0
    V[4][22] = 1.0
    W[4][22] = 1.0
    
    # S2 = a12 * b25
    U[1][23] = 1.0
    V[9][23] = 1.0
    W[4][23] = 1.0
    
    # S3 = a21 * b15
    U[3][24] = 1.0
    V[4][24] = 1.0
    W[9][24] = 1.0
    
    # S4 = a22 * b25
    U[4][25] = 1.0
    V[9][25] = 1.0
    W[9][25] = 1.0
    
    # O1 = a13 * b35
    U[2][26] = 1.0
    V[14][26] = 1.0
    W[4][26] = 1.0
    
    # O2 = a23 * b35
    U[5][27] = 1.0
    V[14][27] = 1.0
    W[9][27] = 1.0

    return {"U": U, "V": V, "W": W}

