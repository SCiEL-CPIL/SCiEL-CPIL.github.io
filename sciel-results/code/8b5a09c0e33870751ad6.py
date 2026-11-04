"""
Tensor decomposition for 4x4 * 4x5 matrix multiplication.

Strategy:
  - Use Strassen's rank-7 decomposition for the 2x2 blocks.
  - Partition the 4x4 * 4x5 multiplication into 2x2 blocks:
    A (4x4) = [A11 A12; A21 A22], each 2x2
    B (4x5) = [B11 B12; B21 B22], B11,B21 are 2x2, B12,B22 are 2x3
  - Each 2x2 * 2x2 product uses rank-7 Strassen.
  - Each 2x2 * 2x3 product can be decomposed using 3 copies of 2x2*2x2? 
    Actually, 2x2 * 2x3 = [2x2 * col1] [2x2 * col2] [2x2 * col3], each 2x2*2x1.
    Alternatively, treat B12 as three 2x1 columns. A 2x2 * 2x1 product can be done with rank 2 (schoolbook for the 1 column) or we can pack.

  Actually, let me think more carefully.
  
  The 4x4 * 4x5 multiplication:
  C (4x5) = A (4x4) * B (4x5)
  
  Partition A into four 2x2 blocks: A11, A12, A21, A22
  Partition B into four blocks: B11 (2x2), B12 (2x3), B21 (2x2), B22 (2x3)
  
  C11 (2x2) = A11*B11 + A12*B21
  C12 (2x3) = A11*B12 + A12*B22
  C21 (2x2) = A21*B11 + A22*B21
  C22 (2x3) = A21*B12 + A22*B22
  
  Each 2x2*2x2 term needs rank 7 (Strassen).
  Each 2x2*2x3 term: A (2x2) * B (2x3). 
  We can decompose 2x2*2x3 by treating B as three 2x1 vectors, but that gives 3 separate 2x2*2x1 products.
  A 2x2 * 2x1 product: result is 2x1. 
  For 2x2 * 2x1, we can use a rank-2 decomposition (schoolbook for one column) or we can use a trick.
  
  Actually, let's just use schoolbook for the 2x2*2x3 parts, or better, note that:
  2x2 * 2x3 can be seen as: for each of the 3 output columns, do a 2x2*2x1.
  A 2x2*2x1 multiplication has rank at most 2 (it's just two dot products).
  So 2x2*2x3 has rank at most 6 (3 columns * 2).
  
  But we can do better. Let me just use schoolbook for all the non-2x2*2x2 parts.
  
  Actually, the simplest approach: use Strassen for the 2x2*2x2 blocks and schoolbook for the rest.
  
  Let me count:
  - 4 terms of 2x2*2x2: each rank 7 → 28
  - 4 terms of 2x2*2x3: each can be done with schoolbook: 2*2*3 = 12 per term → 48
  Total: 28 + 48 = 76.
  
  But wait, we can do 2x2*2x3 more efficiently.
  
  For 2x2 * 2x3:
  A = [a b; c d], B = [e f g; h i j]
  Result = [ae+bg, af+bh, ag+bj; ce+dg, cf+dh, cg+dj]
  
  We can use 6 rank-1 terms (one for each output entry's two contributions... no, that's schoolbook rank 12).
  
  Actually, the tensor rank of 2x2*2x3 is at most 6. We can achieve rank 6:
  For each output column k (k=0,1,2), we have a 2x2*2x1 product.
  A 2x2*2x1: [a b; c d] * [x; y] = [ax+by, cx+dy]
  This can be done with rank 2: 
    term1: a*x, term2: b*y, term3: c*x, term4: d*y → that's rank 4 for one column? No.
    Actually for one column: result = [a*x + b*y, c*x + d*y]
    This is two dot products. We can do it with 2 rank-1 terms if we allow the W to be structured, but in the tensor decomposition framework, each rank-1 term is U_i * V_j * W_k.
    
  Hmm, let me just use the straightforward approach. For 2x2*2x3, I'll use 6 rank-1 terms:
  For column 0: (a*e + b*h, c*e + d*h) → 4 terms in schoolbook
  That's too many.
  
  Let me just use schoolbook for everything and get a correct answer first, then optimize.
  Actually, the task says "rank-minimal", but the score is just the rank. So any valid decomposition with lower rank scores better.
  
  Let me implement:
  1. Strassen for 2x2*2x2 (rank 7 each)
  2. Schoolbook for 2x2*2x3 (rank 12 each)
  Total: 4*7 + 4*12 = 28 + 48 = 76.
  
  This is already better than 80.
  
  Let me implement Strassen properly.
"""

import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def _strassen_decomp_2x2_2x2():
    """
    Returns U, V, W for 2x2 * 2x2 using Strassen's rank-7 method.
    A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen formulas:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U rows are A linear forms, V rows are B linear forms, W columns are C coefficients
    # U[i, k] = coefficient of A_i in M_k's A-linear-form
    # V[j, k] = coefficient of B_j in M_k's B-linear-form
    # W[m, k] = coefficient of M_k in C_m
    
    # M1: A-form = a+d = [1,0,0,1], B-form = e+h = [1,0,0,1]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C11 and C22 get +M1
    
    # M2: A-form = c+d = [0,0,1,1], B-form = e = [1,0,0,0]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C21 gets +M2
    W[3][1] = -1  # C22 gets -M2
    
    # M3: A-form = a = [1,0,0,0], B-form = f-h = [0,1,0,-1]
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C12 gets +M3
    W[3][2] = 1  # C22 gets +M3
    
    # M4: A-form = d = [0,0,0,1], B-form = g-e = [0,0,1,-1]
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C11 gets +M4
    W[2][3] = 1  # C21 gets +M4
    
    # M5: A-form = a+b = [1,1,0,0], B-form = h = [0,0,0,1]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1  # C11 gets -M5
    W[1][4] = 1   # C12 gets +M5
    
    # M6: A-form = c-a = [-1,0,1,0], B-form = e+f = [1,1,0,0]
    U[0][5] = -1; U[2][5] = 1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1   # C22 gets +M6
    
    # M7: A-form = b-d = [0,1,0,-1], B-form = g+h = [0,0,1,1]
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1   # C11 gets +M7
    
    return U, V, W

def _schoolbook_2x2_2x3():
    """
    Returns U, V, W for 2x2 * 2x3 using schoolbook.
    A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    B indices: 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,0), 4=(1,1), 5=(1,2)
    C indices: 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,0), 4=(1,1), 5=(1,2)
    Rank = 2*2*3 = 12
    """
    rank = 12
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(6)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*2+j][col] = 1
                V[j*3+k][col] = 1
                W[i*3+k][col] = 1
                col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We'll build the full decomposition by combining block decompositions.
    # A is 4x4, indexed as A[i, j] with i in 0..3, j in 0..3
    # B is 4x5, indexed as B[j, k] with j in 0..3, k in 0..4
    # C is 4x5, indexed as C[i, k] with i in 0..3, k in 0..4
    
    # Partition:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:5]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:5]
    
    # C11 = A11*B11 + A12*B21  (2x2)
    # C12 = A11*B12 + A12*B22  (2x3)
    # C21 = A21*B11 + A22*B21  (2x2)
    # C22 = A21*B12 + A22*B22  (2x3)
    
    # Each 2x2*2x2 uses Strassen (rank 7)
    # Each 2x2*2x3 uses schoolbook (rank 12)
    
    # Total rank = 4*7 + 4*12 = 28 + 48 = 76
    
    total_rank = 76
    U = [[0]*total_rank for _ in range(n*m)]  # 16 x 76
    V = [[0]*total_rank for _ in range(m*p)]  # 20 x 76
    W = [[0]*total_rank for _ in range(p*n)]  # 20 x 76
    
    col_offset = 0
    
    # Helper to add a block decomposition
    def add_block(U_block, V_block, W_block, a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col_offset
        r = len(U_block[0])
        for ci in range(r):
            # U_block[i, ci] -> U[a_row_offset + i, col_offset + ci]
            for i in range(len(U_block)):
                U[a_row_offset + i][col_offset + ci] += U_block[i][ci]
            for j in range(len(V_block)):
                V[b_row_offset + j][col_offset + ci] += V_block[j][ci]
            for k in range(len(W_block)):
                W[c_row_offset + k][col_offset + ci] += W_block[k][ci]
        col_offset += r
    
    # Get Strassen decomposition
    U_s, V_s, W_s = _strassen_decomp_2x2_2x2()
    # Get schoolbook 2x2*2x3 decomposition
    U_sc, V_sc, W_sc = _schoolbook_2x2_2x3()
    
    # C11 = A11*B11 + A12*B21
    # A11: rows 0-1, cols 0-1 → a_row_offset=0, a_col_offset=0
    # B11: rows 0-1, cols 0-1 → b_row_offset=0, b_col_offset=0
    # C11: rows 0-1, cols 0-1 → c_row_offset=0, c_col_offset=0
    add_block(U_s, V_s, W_s, 0, 0, 0, 0, 0, 0)
    
    # A12: rows 0-1, cols 2-3 → a_row_offset=0, a_col_offset=4 (since m=4, col 2 → index 2*1+... wait)
    # A is 4x4, flattened as A[i*m+j] = A[i*4+j]
    # A12: i in 0..1, j in 2..3 → indices 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # In the 2x2 block, local indices 0,1,2,3 map to global 2,3,6,7
    # a_row_offset for the block: the first local row 0 maps to global row 0, col 2 → global index 2
    # But our add_block assumes contiguous rows. Let me handle this differently.
    
    # Actually, I need to be more careful. Let me just manually map.
    
    # Let me restart with a cleaner approach.
    
    # I'll build the U, V, W matrices by iterating over each term.
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col = 0
    
    # Define helper: for a Strassen term, add it with given offsets
    def add_strassen(a_base_i, a_base_j, b_base_j, b_base_k, c_base_i, c_base_k):
        nonlocal col
        for ci in range(7):
            # U: A linear forms
            for li in range(4):
                val = U_s[li][ci]
                if val != 0:
                    gi = a_base_i + li // 2
                    gj = a_base_j + li % 2
                    U[gi*m + gj][col] += val
            # V: B linear forms
            for lj in range(4):
                val = V_s[lj][ci]
                if val != 0:
                    gj = b_base_j + lj // 2
                    gk = b_base_k + lj % 2
                    V[gj*p + gk][col] += val
            # W: C coefficients
            for lk in range(4):
                val = W_s[lk][ci]
                if val != 0:
                    gi = c_base_i + lk // 2
                    gk = c_base_k + lk % 2
                    W[gk*n + gi][col] += val
        col += 7
    
    def add_schoolbook_2x2_2x3(a_base_i, a_base_j, b_base_j, b_base_k, c_base_i, c_base_k):
        nonlocal col
        for ci in range(12):
            # U: A linear forms (2x2)
            li = ci % 4
            gi = a_base_i + li // 2
            gj = a_base_j + li % 2
            U[gi*m + gj][col] += U_sc[li][ci]
            
            # V: B linear forms (2x3)
            lj = (ci // 4) % 3
            gj = b_base_j + (ci // 12) * 0 + 0  # This is getting complicated
            # Actually, for 2x2*2x3 schoolbook:
            # ci = i*2*3 + j*3 + k, where i in 0..1, j in 0..1, k in 0..2
            i_local = ci // 6
            j_local = (ci % 6) // 3
            k_local = ci % 3
            gi = a_base_i + i_local
            gj = a_base_j + j_local
            U[gi*m + gj][col] += U_sc[i_local*2 + j_local][ci]
            
            gj = b_base_j + j_local
            gk = b_base_k + k_local
            V[gj*p + gk][col] += V_sc[j_local*3 + k_local][ci]
            
            gi = c_base_i + i_local
            gk = c_base_k + k_local
            W[gk*n + gi][col] += W_sc[i_local*3 + k_local][ci]
        col += 12
    
    # C11 = A11*B11: A11 at (0,0), B11 at (0,0), C11 at (0,0)
    add_strassen(0, 0, 0, 0, 0, 0)
    
    # C11 += A12*B21: A12 at (0,2), B21 at (2,0), C11 at (0,0)
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # C12 = A11*B12: A11 at (0,0), B12 at (0,2), C12 at (0,2)
    add_schoolbook_2x2_2x3(0, 0, 0, 2, 0, 2)
    
    # C12 += A12*B22: A12 at (0,2), B22 at (2,2), C12 at (0,2)
    add_schoolbook_2x2_2x3(0, 2, 2, 2, 0, 2)
    
    # C21 = A21*B11: A21 at (2,0), B11 at (0,0), C21 at (2,0)
    add_strassen(2, 0, 0, 0, 2, 0)
    
    # C21 += A22*B21: A22 at (2,2), B21 at (2,0), C21 at (2,0)
    add_strassen(2, 2, 2, 0, 2, 0)
    
    # C22 = A21*B12: A21 at (2,0), B12 at (0,2), C22 at (2,2)
    add_schoolbook_2x2_2x3(2, 0, 0, 2, 2, 2)
    
    # C22 += A22*B22: A22 at (2,2), B22 at (2,2), C22 at (2,2)
    add_schoolbook_2x2_2x3(2, 2, 2, 2, 2, 2)
    
    return {"U": U, "V": V, "W": W}
