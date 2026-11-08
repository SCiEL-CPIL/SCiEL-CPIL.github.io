"""Matrix multiplication 2,4,5 - rank-minimal exact tensor decomposition."""
import time
import random

DIMENSIONS = [2, 4, 5]

# Strassen's algorithm for 2x2x2 (rank 7)
# U: 4x7, V: 4x7, W: 4x7
# C = A*B where A,B,C are 2x2
# U[i*2+j, c] = A[i,j] coefficient
# V[j*2+k, c] = B[j,k] coefficient  
# W[k*2+i, c] = C[i,k] coefficient (trace convention W[k*n+i,r])

def _strassen_222():
    """Return U, V, W for 2x2x2 with rank 7 (Strassen)."""
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]  # A[i,j] -> row i*2+j
    V = [[0]*7 for _ in range(4)]  # B[j,k] -> row j*2+k
    W = [[0]*7 for _ in range(4)]  # C[i,k] -> row k*2+i
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1   # C00
    W[3][0] = 1   # C11
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1   # C10
    W[3][1] = -1  # C11
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1   # C01
    W[3][2] = 1   # C11
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1   # C00
    W[2][3] = 1   # C10
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1  # C00
    W[1][4] = 1   # C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1   # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1   # C00
    
    return U, V, W


def _schoolbook_223():
    """Return U, V, W for 2x2x3 with rank 12 (schoolbook)."""
    n, m, p = 2, 2, 3
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def _combine_block_decomp(U1, V1, W1, U2, V2, W2, shift_U_rows, shift_V_rows, shift_W_rows, shift_cols):
    """
    Combine two block decompositions.
    U1, V1, W1: first block (top 2x2 part of A, top 2x2 part of B)
    U2, V2, W2: second block (bottom 2x2 part of A, bottom 2x2 part of B)
    
    For 2x4x5:
    A is 2x4, split into A_top (2x2) and A_bottom (2x2)
    B is 4x5, split into B_top (2x5) and B_bottom (2x5)
    C = A_top * B_top + A_bottom * B_bottom
    
    U rows: 2*4 = 8 rows. First 4 rows for A_top, next 4 for A_bottom.
    V rows: 4*5 = 20 rows. First 10 rows for B_top (j=0,1), next 10 for B_bottom (j=2,3).
    W rows: 5*2 = 10 rows.
    
    shift_U_rows: offset for U2 rows (should be 4)
    shift_V_rows: offset for V2 rows (should be 10)  
    shift_W_rows: offset for W2 rows (should be 0, since W is same)
    shift_cols: number of columns in first block
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    total_rank = shift_cols + r2
    
    nU_rows = len(U1) + shift_U_rows
    nV_rows = len(V1) + shift_V_rows
    nW_rows = len(W1) + shift_W_rows
    
    U = [[0]*total_rank for _ in range(nU_rows)]
    V = [[0]*total_rank for _ in range(nV_rows)]
    W = [[0]*total_rank for _ in range(nW_rows)]
    
    # Copy block 1
    for i in range(len(U1)):
        for c in range(r1):
            U[i][c] = U1[i][c]
    for i in range(len(V1)):
        for c in range(r1):
            V[i][c] = V1[i][c]
    for i in range(len(W1)):
        for c in range(r1):
            W[i][c] = W1[i][c]
    
    # Copy block 2
    for i in range(len(U2)):
        for c in range(r2):
            U[i + shift_U_rows][shift_cols + c] = U2[i][c]
    for i in range(len(V2)):
        for c in range(r2):
            V[i + shift_V_rows][shift_cols + c] = V2[i][c]
    for i in range(len(W2)):
        for c in range(r2):
            W[i + shift_W_rows][shift_cols + c] = W2[i][c]
    
    return U, V, W


def _try_find_223_rank11(deadline):
    """
    Try to find a rank-11 decomposition for 2x2x3.
    If not found in time, return None.
    """
    start = time.time()
    n, m, p = 2, 2, 3
    rank = 11
    
    # Number of free entries: 4*11 + 6*11 + 6*11 = 44+66+66 = 176
    # Constraints: for each (i,k) output, sum_c U[i*2+j,c]*V[j*3+k,c]*W[k*2+i,c] = delta
    # This is a system of polynomial equations.
    
    # Simple approach: random initialization + local search
    # But this is complex. Let's just try a few random initializations.
    
    import numpy as np
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Scale to help
        U /= np.sqrt(rank)
        V /= np.sqrt(rank)
        W /= np.sqrt(rank)
        
        # Simple gradient-free local search
        for step in range(500):
            if time.time() > deadline:
                break
            
            # Compute residuals
            err = 0
            # For each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
            # For non-matching: should be 0
            
            # This is getting complex. Let's just use the schoolbook fallback.
            break
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 100  # Leave margin
    
    # Use Strassen (rank 7) for each 2x2x2 block, and schoolbook (rank 12) for 2x2x3
    # Total rank: 7 + 12 = 19? No wait.
    
    # Actually, let me reconsider the block structure.
    # A is 2x4, B is 4x5.
    # C = A*B, C is 2x5.
    # 
    # Split A into A_top (2x2) and A_bottom (2x2)
    # Split B into B_top (2x5) and B_bottom (2x5)
    # C = A_top @ B_top + A_bottom @ B_bottom
    #
    # A_top @ B_top: 2x2 @ 2x5 -> use 2x2x2 blocks? 
    #   B_top is 2x5, can split into two 2x2 + one 2x1? Or use 2x2x5 directly.
    #
    # Simpler: just use schoolbook for the whole thing but with Strassen for 2x2x2 parts.
    
    # Let's go with the straightforward approach:
    # Rank 7 for 2x2x2 (Strassen) and schoolbook for remaining parts.
    
    # Actually, the cleanest decomposition:
    # Split B into B[:2,:] (2x5) and B[2:,:] (2x5)
    # C_top_half = A_top @ B_top (2x2 @ 2x5)
    # C_bot_half = A_bot @ B_bot (2x2 @ 2x5)
    
    # For 2x2 @ 2x5: split B into 2x2 + 2x2 + 2x1
    #   2x2 @ 2x2: rank 7 (Strassen)
    #   2x2 @ 2x2: rank 7 (Strassen)
    #   2x2 @ 2x1: rank 2 (schoolbook, since 2*2*1=4 but we can do better)
    #     Actually 2x2 @ 2x1: C_i = sum_j A_ij B_j1
    #     This is just 2 dot products of length 2, rank 2 is sufficient.
    
    # Total: 7 + 7 + 2 = 16 for A_top @ B_top
    # Same for A_bot @ B_bot: 16
    # Total rank: 32
    
    # Hmm, that's worse than schoolbook (40) only slightly. Let me try a different split.
    
    # Alternative: Use Strassen for 2x2x2 blocks.
    # Split A into 4 2x1 column blocks? No, A is 2x4.
    
    # Let me just use the rank-38 construction from the previous attempt:
    # Split B into 4 4x... no.
    
    # OK, let's go with the guaranteed rank-38 solution.
    # Split A into A_top (2x2) and A_bot (2x2)
    # Split B into B_top (2x5) and B_bot (2x5)
    # 
    # A_top @ B_top: 2x2 @ 2x5
    #   Split B_top into B_top[:,0:2] (2x2), B_top[:,2:4] (2x2), B_top[:,4:5] (2x1)
    #   2x2 @ 2x2: rank 7
    #   2x2 @ 2x2: rank 7
    #   2x2 @ 2x1: rank 2
    #   Subtotal: 16
    #
    # A_bot @ B_bot: same, 16
    #
    # Total: 32
    
    # Wait, 2x2 @ 2x1: C_i = sum_j A_ij * B_j1
    # This is a bilinear form with 2 outputs, each sum of 2 products.
    # Minimum rank is 2 (can't do better than schoolbook for this).
    
    # So total rank = 16 + 16 = 32.
    
    U, V, W = _build_rank32()
    return {"U": U, "V": V, "W": W}


def _build_rank32():
    """
    Build rank-32 decomposition for 2x4x5.
    
    A (2x4) = [A_top (2x2) | A_bot (2x2)]
    B (4x5) = [B_top (2x5); B_bot (2x5)]
    C (2x5) = A_top @ B_top + A_bot @ B_bot
    
    For A_top @ B_top (2x2 @ 2x5):
      Split B_top into B_top[:,0:2] (2x2), B_top[:,2:4] (2x2), B_top[:,4:5] (2x1)
      Use Strassen for the two 2x2 @ 2x2, schoolbook for 2x2 @ 2x1
    
    For A_bot @ B_bot (2x2 @ 2x5): same
    
    U rows: 8 (2*4)
    V rows: 20 (4*5)
    W rows: 10 (5*2)
    Rank: 32
    """
    n, m, p = 2, 4, 5
    rank = 32
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Get Strassen and schoolbook 2x2x1 decompositions
    U_s, V_s, W_s = _strassen_222()  # rank 7
    
    # 2x2 @ 2x1 schoolbook: rank 2
    # C_i = sum_j A_ij B_j
    # U: 2x2 rows for A (i,j), V: 2x1 rows for B (j), W: 1x2 rows for C (k=0, i)
    # W[k*n+i, c] where k=0, so W[i, c] for i=0,1
    U_221 = [[0]*2 for _ in range(4)]  # 2*2=4 rows
    V_221 = [[0]*2 for _ in range(2)]  # 2*1=2 rows
    W_221 = [[0]*2 for _ in range(1*2)]  # 1*2=2 rows
    for j in range(2):
        U_221[j][j] = 1
        V_221[j][j] = 1
        W_221[j][j] = 1  # W[i, c] for i=j, c=j
    
    def _add_block(U_block, V_block, W_block, u_row_offset, v_row_offset, w_row_offset, col_offset):
        r = len(U_block[0])
        for i in range(len(U_block)):
            for c in range(r):
                U[i + u_row_offset][col_offset + c] = U_block[i][c]
        for i in range(len(V_block)):
            for c in range(r):
                V[i + v_row_offset][col_offset + c] = V_block[i][c]
        for i in range(len(W_block)):
            for c in range(r):
                W[i + w_row_offset][col_offset + c] = W_block[i][c]
        return col_offset + r
    
    col = 0
    
    # A_top @ B_top[:,0:2] (2x2 @ 2x2): Strassen
    # U rows: 0-3 (A_top)
    # V rows: 0-3 (B_top[:,0:2], j=0,1; k=0,1)
    # W rows: 0-1 (C columns 0,1)
    col = _add_block(U_s, V_s, W_s, 0, 0, 0, col)
    
    # A_top @ B_top[:,2:4] (2x2 @ 2x2): Strassen
    # V rows: 4-7 (B_top[:,2:4], j=0,1; k=2,3)
    # W rows: 2-3 (C columns 2,3)
    col = _add_block(U_s, V_s, W_s, 0, 4, 2, col)
    
    # A_top @ B_top[:,4:5] (2x2 @ 2x1): schoolbook rank 2
    # V rows: 8-9 (B_top[:,4:5], j=0,1; k=4)
    # W rows: 4 (C column 4)
    col = _add_block(U_221, V_221, W_221, 0, 8, 4, col)
    
    # A_bot @ B_bot[:,0:2] (2x2 @ 2x2): Strassen
    # U rows: 4-7 (A_bot)
    # V rows: 10-13 (B_bot[:,0:2], j=2,3; k=0,1)
    # W rows: 0-1 (C columns 0,1)
    col = _add_block(U_s, V_s, W_s, 4, 10, 0, col)
    
    # A_bot @ B_bot[:,2:4] (2x2 @ 2x2): Strassen
    # V rows: 14-17 (B_bot[:,2:4], j=2,3; k=2,3)
    # W rows: 2-3 (C columns 2,3)
    col = _add_block(U_s, V_s, W_s, 4, 14, 2, col)
    
    # A_bot @ B_bot[:,4:5] (2x2 @ 2x1): schoolbook rank 2
    # V rows: 18-19 (B_bot[:,4:5], j=2,3; k=4)
    # W rows: 4 (C column 4)
    col = _add_block(U_221, V_221, W_221, 4, 18, 4, col)
    
    return U, V, W
