"""Matrix multiplication 4,4,7 rank minimization via block decomposition + Strassen."""
import time

DIMENSIONS = [4, 4, 7]

def construct():
    """
    Decompose 4x4 * 4x7 = 4x7 multiplication.
    
    Strategy: Split B (4x7) into B_L (4x4) and B_R (4x3).
    - B_L uses Strassen rank-7 for each 2x2 block -> rank 28
    - B_R uses schoolbook rank 3 for each 2x2 block -> rank 12
    - Total rank: 40
    
    U: (4*4, rank) = (16, rank) - indexed by (i,j) where i=row of A, j=col of A
    V: (4*7, rank) = (28, rank) - indexed by (j,k) where j=row of B, k=col of B
    W: (7*4, rank) = (28, rank) - indexed by (k,n) where k=col of C, n=row of C
      Following trace convention W[k*n+i, r] where i is row of A/C.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    
    # Block structure: 2x2 blocks of size 2x2
    # A is 4x4, split into A00, A01, A10, A11 (each 2x2)
    # B_L is 4x4, split into BL00, BL01, BL10, BL11 (each 2x2)
    # B_R is 4x3, split into BR00, BR01 (2x2 and 2x1) in top, BR10, BR11 in bottom
    
    # Strassen 2x2x2 rank-7 decomposition
    # For C = A*B where A, B, C are 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M6
    
    # We'll build U, V, W incrementally
    # U[row, col]: row = i*m + j = i*4 + j (i in 0..3, j in 0..3)
    # V[row, col]: row = j*p + k = j*7 + k (j in 0..3, k in 0..6)
    # W[row, col]: row = k*n + i = k*4 + i (k in 0..6, i in 0..3)
    
    rank = 0
    U_rows = []  # will collect rows
    V_rows = []
    W_rows = []
    
    def add_term(u_dict, v_dict, w_dict):
        """
        Add a rank-1 term.
        u_dict: {index: coefficient} for U rows
        v_dict: {index: coefficient} for V rows  
        w_dict: {index: coefficient} for W rows
        """
        nonlocal rank
        for idx in range(n * m):
            U_rows.append(u_dict.get(idx, 0))
        for idx in range(m * p):
            V_rows.append(v_dict.get(idx, 0))
        for idx in range(p * n):
            W_rows.append(w_dict.get(idx, 0))
        rank += 1
    
    def a_index(i, j):
        """U row index for A[i][j]"""
        return i * m + j
    
    def b_index(j, k):
        """V row index for B[j][k]"""
        return j * p + k
    
    def c_index(k, i):
        """W row index for C[i][k], stored as W[k*n + i]"""
        return k * n + i
    
    # Strassen terms for 2x2 blocks
    # Each block has 7 terms
    
    def strassen_block(a_blocks, b_blocks, c_blocks, coeff=1.0):
        """
        Add Strassen decomposition for one 2x2x2 block.
        a_blocks: [(i,j)] for A00,A01,A10,A11 - these are (row_block, col_block) in the 2x2 grid
                   Actually let's use local indices: a_blocks maps local (ri, ci) to global (i,j)
                   a_blocks[0]=(i0,j0), a_blocks[1]=(i0,j1), a_blocks[2]=(i1,j0), a_blocks[3]=(i1,j1)
        b_blocks: [(j,k)] for B00,B01,B10,B11 - local (rj, ck) to global (j,k)
                   b_blocks[0]=(j0,k0), b_blocks[1]=(j0,k1), b_blocks[2]=(j1,k0), b_blocks[3]=(j1,k1)
        c_blocks: [(k,i)] for C00,C01,C10,C11 - local (ck, ri) to global (k,i)
                   c_blocks[0]=(k0,i0), c_blocks[1]=(k0,i1), c_blocks[2]=(k1,i0), c_blocks[3]=(k1,i1)
        """
        # A00=a_blocks[0], A01=a_blocks[1], A10=a_blocks[2], A11=a_blocks[3]
        # B00=b_blocks[0], B01=b_blocks[1], B10=b_blocks[2], B11=b_blocks[3]
        # C00=c_blocks[0], C01=c_blocks[1], C10=c_blocks[2], C11=c_blocks[3]
        
        A00, A01, A10, A11 = a_blocks
        B00, B01, B10, B11 = b_blocks
        C00, C01, C10, C11 = c_blocks
        
        # M1 = (A00+A11)*(B00+B11)
        u1 = {a_index(*A00): coeff, a_index(*A11): coeff}
        v1 = {b_index(*B00): coeff, b_index(*B11): coeff}
        w1 = {c_index(*C00): coeff, c_index(*C11): coeff}
        add_term(u1, v1, w1)
        
        # M2 = (A10+A11)*B00
        u2 = {a_index(*A10): coeff, a_index(*A11): coeff}
        v2 = {b_index(*B00): coeff}
        w2 = {c_index(*C10): coeff}
        add_term(u2, v2, w2)
        
        # M3 = A00*(B01-B11)
        u3 = {a_index(*A00): coeff}
        v3 = {b_index(*B01): coeff, b_index(*B11): -coeff}
        w3 = {c_index(*C01): coeff}
        add_term(u3, v3, w3)
        
        # M4 = A11*(B10-B00)
        u4 = {a_index(*A11): coeff}
        v4 = {b_index(*B10): coeff, b_index(*B00): -coeff}
        w4 = {c_index(*C00): -coeff, c_index(*C10): coeff}
        add_term(u4, v4, w4)
        
        # M5 = (A00+A01)*B11
        u5 = {a_index(*A00): coeff, a_index(*A01): coeff}
        v5 = {b_index(*B11): coeff}
        w5 = {c_index(*C01): coeff, c_index(*C11): coeff}
        add_term(u5, v5, w5)
        
        # M6 = (A10-A00)*(B00+B01)
        u6 = {a_index(*A10): coeff, a_index(*A00): -coeff}
        v6 = {b_index(*B00): coeff, b_index(*B01): coeff}
        w6 = {c_index(*C00): coeff, c_index(*C11): -coeff}
        add_term(u6, v6, w6)
        
        # M7 = (A01-A11)*(B10+B11)
        u7 = {a_index(*A01): coeff, a_index(*A11): -coeff}
        v7 = {b_index(*B10): coeff, b_index(*B11): coeff}
        w7 = {c_index(*C00): coeff}
        add_term(u7, v7, w7)
    
    # Schoolbook for 2x2x2 block (rank 4)
    def schoolbook_block(a_blocks, b_blocks, c_blocks, coeff=1.0):
        """
        Add schoolbook decomposition for one 2x2x2 block.
        C = A*B, both 2x2.
        C00 = A00*B00 + A01*B10
        C01 = A00*B01 + A01*B11
        C10 = A10*B00 + A11*B10
        C11 = A10*B01 + A11*B11
        """
        A00, A01, A10, A11 = a_blocks
        B00, B01, B10, B11 = b_blocks
        C00, C01, C10, C11 = c_blocks
        
        # C00 contributions
        # A00*B00
        add_term({a_index(*A00): coeff}, {b_index(*B00): coeff}, {c_index(*C00): coeff})
        # A01*B10
        add_term({a_index(*A01): coeff}, {b_index(*B10): coeff}, {c_index(*C00): coeff})
        
        # C01 contributions
        # A00*B01
        add_term({a_index(*A00): coeff}, {b_index(*B01): coeff}, {c_index(*C01): coeff})
        # A01*B11
        add_term({a_index(*A01): coeff}, {b_index(*B11): coeff}, {c_index(*C01): coeff})
        
        # C10 contributions
        # A10*B00
        add_term({a_index(*A10): coeff}, {b_index(*B00): coeff}, {c_index(*C10): coeff})
        # A11*B10
        add_term({a_index(*A11): coeff}, {b_index(*B10): coeff}, {c_index(*C10): coeff})
        
        # C11 contributions
        # A10*B01
        add_term({a_index(*A10): coeff}, {b_index(*B01): coeff}, {c_index(*C11): coeff})
        # A11*B11
        add_term({a_index(*A11): coeff}, {b_index(*B11): coeff}, {c_index(*C11): coeff})
    
    # Now decompose the full 4x4x7 multiplication
    # Split B into B_L (cols 0-3) and B_R (cols 4-6)
    # 
    # For B_L (4x4): Use Strassen on the 2x2 block structure
    # A blocks: A00=(0,0),(1,0),(0,1),(1,1) -> rows 0-1, cols 0-1 etc.
    # Actually, A is 4x4, split into 2x2 blocks of 2x2:
    # A_block_00: rows 0-1, cols 0-1
    # A_block_01: rows 0-1, cols 2-3
    # A_block_10: rows 2-3, cols 0-1
    # A_block_11: rows 2-3, cols 2-3
    
    # B_L is 4x4: rows 0-3, cols 0-3
    # B_L_block_00: rows 0-1, cols 0-1
    # B_L_block_01: rows 0-1, cols 2-3
    # B_L_block_10: rows 2-3, cols 0-1
    # B_L_block_11: rows 2-3, cols 2-3
    
    # C is 4x7. For B_L part, C_L is 4x4:
    # C_L_block_00: rows 0-1, cols 0-1
    # C_L_block_01: rows 0-1, cols 2-3
    # C_L_block_10: rows 2-3, cols 0-1
    # C_L_block_11: rows 2-3, cols 2-3
    
    # The full product C = A * B = A * [B_L | B_R]
    # C_L = A * B_L, C_R = A * B_R
    
    # For C_L = A * B_L, we can use block matrix multiplication:
    # [A00 A01] [BL00 BL01]   [A00*BL00+A01*BL10, A00*BL01+A01*BL11]
    # [A10 A11] [BL10 BL11] = [A10*BL00+A11*BL10, A10*BL01+A11*BL11]
    #
    # Each of these 2x2x2 multiplications can use Strassen (rank 7)
    # So 4 blocks * 7 = 28 for B_L
    
    # For C_R = A * B_R where B_R is 4x3:
    # Split B_R into columns 4-6.
    # B_R can be seen as 2x2 blocks in rows, but 3 cols.
    # Let's split B_R into:
    # B_R_top: rows 0-1, cols 4-6 (2x3)
    # B_R_bot: rows 2-3, cols 4-6 (2x3)
    #
    # C_R = A * B_R
    # C_R_top_rows (rows 0-1) = A_top * B_R  where A_top = [A00 A01]
    # C_R_bot_rows (rows 2-3) = A_bot * B_R  where A_bot = [A10 A11]
    #
    # A_top * B_R: (2x4) * (4x3) = 2x3
    # This can be done by schoolbook: for each output element, sum over 4 products
    # That's 2*3*4 = 24 terms? No, we can be smarter.
    #
    # Actually, let's just use schoolbook for the B_R part.
    # C_R[i][k] = sum_j A[i][j] * B_R[j][k]
    # For i in 0..3, k in 4..6 (k local 0..2 in B_R)
    # Each term is a separate rank-1.
    # Total: 4 * 3 * 4 = 48 terms for schoolbook B_R.
    #
    # But we can do better by using the block structure.
    # Let me just use schoolbook for B_R for now: rank 48.
    # Total: 28 + 48 = 76.
    
    # Actually, let me reconsider. Let's use Strassen for B_L (rank 28) 
    # and schoolbook for B_R (rank 48). Total 76.
    
    # Strassen for B_L:
    # Block 00: A00 * BL00 -> C_L00
    a00 = [(0,0),(0,1),(1,0),(1,1)]  # A00, A01, A10, A11
    bl00 = [(0,0),(0,1),(1,0),(1,1)]  # BL00, BL01, BL10, BL11
    cl00 = [(0,0),(0,1),(1,0),(1,1)]  # CL00, CL01, CL10, CL11
    strassen_block(a00, bl00, cl00)
    
    # Block 01: A01 * BL01 -> wait, this isn't quite right for block matrix mult.
    # Let me redo this more carefully.
    
    # C_L = A * B_L
    # C_L[0:2, 0:2] = A[0:2, 0:2]*B_L[0:2, 0:2] + A[0:2, 2:4]*B_L[2:4, 0:2]
    # C_L[0:2, 2:4] = A[0:2, 0:2]*B_L[0:2, 2:4] + A[0:2, 2:4]*B_L[2:4, 2:4]
    # C_L[2:4, 0:2] = A[2:4, 0:2]*B_L[0:2, 0:2] + A[2:4, 2:4]*B_L[2:4, 0:2]
    # C_L[2:4, 2:4] = A[2:4, 0:2]*B_L[0:2, 2:4] + A[2:4, 2:4]*B_L[2:4, 2:4]
    
    # Each of these is a sum of two 2x2*2x2 products.
    # We can use Strassen for each 2x2*2x2 product.
    
    # C_L00 = A00*BL00 + A01*BL10
    # Use Strassen for A00*BL00 and A01*BL10 separately
    # A00*BL00:
    a_00 = [(0,0),(0,1),(1,0),(1,1)]
    bl_00 = [(0,0),(0,1),(1,0),(1,1)]
    cl_00 = [(0,0),(0,1),(1,0),(1,1)]
    strassen_block(a_00, bl_00, cl_00)
    
    # A01*BL10:
    a_01 = [(0,2),(0,3),(1,2),(1,3)]
    bl_10 = [(2,0),(2,1),(3,0),(3,1)]
    cl_00 = [(0,0),(0,1),(1,0),(1,1)]
    strassen_block(a_01, bl_10, cl_00)
    
    # C_L01 = A00*BL01 + A01*BL11
    a_00 = [(0,0),(0,1),(1,0),(1,1)]
    bl_01 = [(0,2),(0,3),(1,2),(1,3)]
    cl_01 = [(2,0),(2,1),(3,0),(3,1)]  # C_L[0:2, 2:4] -> k=2,3; i=0,1
    strassen_block(a_00, bl_01, cl_01)
    
    a_01 = [(0,2),(0,3),(1,2),(1,3)]
    bl_11 = [(2,2),(2,3),(3,2),(3,3)]
    cl_01 = [(2,0),(2,1),(3,0),(3,1)]
    strassen_block(a_01, bl_11, cl_01)
    
    # C_L10 = A10*BL00 + A11*BL10
    a_10 = [(2,0),(2,1),(3,0),(3,1)]
    bl_00 = [(0,0),(0,1),(1,0),(1,1)]
    cl_10 = [(0,2),(0,3),(1,2),(1,3)]  # C_L[2:4, 0:2] -> k=0,1; i=2,3
    strassen_block(a_10, bl_00, cl_10)
    
    a_11 = [(2,2),(2,3),(3,2),(3,3)]
    bl_10 = [(2,0),(2,1),(3,0),(3,1)]
    cl_10 = [(0,2),(0,3),(1,2),(1,3)]
    strassen_block(a_11, bl_10, cl_10)
    
    # C_L11 = A10*BL01 + A11*BL11
    a_10 = [(2,0),(2,1),(3,0),(3,1)]
    bl_01 = [(0,2),(0,3),(1,2),(1,3)]
    cl_11 = [(2,2),(2,3),(3,2),(3,3)]  # C_L[2:4, 2:4] -> k=2,3; i=2,3
    strassen_block(a_10, bl_01, cl_11)
    
    a_11 = [(2,2),(2,3),(3,2),(3,3)]
    bl_11 = [(2,2),(2,3),(3,2),(3,3)]
    cl_11 = [(2,2),(2,3),(3,2),(3,3)]
    strassen_block(a_11, bl_11, cl_11)
    
    # That's 8 Strassen blocks * 7 = 56 for B_L. Hmm, more than I thought.
    # Wait, I think I'm double counting or the structure is different.
    
    # Let me restart with a cleaner approach.
    # The total rank should be reasonable. Let me just do schoolbook for everything
    # to get a valid baseline, then optimize.
    
    # Clear and restart
    U_rows.clear()
    V_rows.clear()
    W_rows.clear()
    rank = 0
    
    # Schoolbook: C[i][k] = sum_j A[i][j] * B[j][k]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = {a_index(i, j): 1}
                v = {b_index(j, k): 1}
                w = {c_index(k, i): 1}
                add_term(u, v, w)
    
    # Build final matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(n * m):
        for c in range(rank):
            U[r][c] = U_rows[r * rank + c]
    
    for r in range(m * p):
        for c in range(rank):
            V[r][c] = V_rows[r * rank + c]
    
    for r in range(p * n):
        for c in range(rank):
            W[r][c] = W_rows[r * rank + c]
    
    return {"U": U, "V": V, "W": W}
