"""
Strategy: Structural Block Decomposition with Strassen for 2x2x2 sub-blocks.

The tensor for multiplying a 3x4 matrix by a 4x8 matrix has dimensions (3, 4, 8).
Schoolbook rank is 3*4*8 = 96.

We can partition the first dimension (3) into 1 and 2.
Let A = [A1; A2] where A1 is 1x4 and A2 is 2x4.
Let B be 4x8.
C = A1*B + A2*B.

The term A1*B involves a 1x4 matrix and a 4x8 matrix.
The tensor for 1x4x8 matrix multiplication has rank 1*4*8 = 32.
This is minimal because for a 1xm matrix, the output is a linear combination of the m rows of B,
and each of the p columns requires m multiplications. Specifically, C1[k] = sum_j A1[j] B[j,k].
Each of the 8 columns of C1 requires 4 multiplications. Total 32. No savings possible for 1xm x mxp.

The term A2*B involves a 2x4 matrix and a 4x8 matrix.
Dimensions: 2, 4, 8.
We can partition the middle dimension (4) into two 2s: B = [B_left, B_right] where B_left is 2x8, B_right is 2x8.
Wait, B is 4x8. A2 is 2x4.
A2 = [A21, A22] where A21 is 2x2 and A22 is 2x2.
B = [B1; B2] where B1 is 2x8 and B2 is 2x8.
Then A2*B = A21*B1 + A22*B2.

Now we have two sub-problems:
1. A21 (2x2) * B1 (2x8)
2. A22 (2x2) * B2 (2x8)

For A21 (2x2) * B1 (2x8):
We can partition B1 into four 2x2 blocks: B1 = [B11 B12; B13 B14] is wrong, B1 is 2x8.
B1 = [B1_col1, B1_col2, B1_col3, B1_col4] where each is 2x2.
So A21 * B1 = A21*B1_c1 + A21*B1_c2 + A21*B1_c3 + A21*B1_c4.
Each term is 2x2 * 2x2.
The rank of 2x2x2 matrix multiplication is 7 (Strassen).
So each term takes rank 7.
There are 4 such terms. Total rank for A21*B1 is 4 * 7 = 28.

Similarly, A22 (2x2) * B2 (2x8) also takes rank 4 * 7 = 28.

Total rank for A2*B = 28 + 28 = 56.
Total rank for C = Rank(A1*B) + Rank(A2*B) = 32 + 56 = 88.

This constructs a valid rank-88 decomposition.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-88 exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x8 matrix.
    
    Structure:
    - Row 0 of A (1x4) multiplied by B (4x8): Rank 32 (Schoolbook).
    - Rows 1-2 of A (2x4) multiplied by B (4x8): Rank 56 (Strassen on 2x2x2 blocks).
    
    Total Rank: 32 + 56 = 88.
    """
    
    n, m, p = 3, 4, 8
    
    # U shape: (n*m, rank) = (12, 88)
    # V shape: (m*p, rank) = (32, 88)
    # W shape: (p*n, rank) = (24, 88)
    
    rank_total = 88
    U = np.zeros((n*m, rank_total), dtype=np.int64)
    V = np.zeros((m*p, rank_total), dtype=np.int64)
    W = np.zeros((p*n, rank_total), dtype=np.int64)
    
    col = 0
    
    # Part 1: A_row0 (1x4) * B (4x8)
    # Indices for A: i=0, j=0..3. U indices: 0..3.
    # Indices for B: j=0..3, k=0..7. V indices: j*8 + k.
    # Indices for C: i=0, k=0..7. W indices: k*3 + 0.
    
    for j in range(4):
        for k in range(8):
            # U[i*m + j] -> U[0*4 + j] = U[j]
            U[j, col] = 1
            # V[j*p + k] -> V[j*8 + k]
            V[j*8 + k, col] = 1
            # W[k*n + i] -> W[k*3 + 0]
            W[k*3 + 0, col] = 1
            col += 1
            
    # Part 2: A_rows12 (2x4) * B (4x8)
    # We split B into two 2x8 halves vertically? No, B is 4x8.
    # A_rows12 is 2x4. Let's call it A2.
    # A2 = [A21 A22] where A21 is 2x2 (cols 0,1) and A22 is 2x2 (cols 2,3).
    # B = [B_top; B_bot] where B_top is 2x8 (rows 0,1) and B_bot is 2x8 (rows 2,3).
    # A2 * B = A21 * B_top + A22 * B_bot.
    
    # We need to decompose A21 (2x2) * B_top (2x8) and A22 (2x2) * B_bot (2x8).
    # B_top is 2x8. We can split it into four 2x2 blocks horizontally.
    # B_top = [BT1 BT2 BT3 BT4]
    # A21 * B_top = A21*BT1 + A21*BT2 + A21*BT3 + A21*BT4.
    # Each is 2x2 * 2x2. Rank 7 each. Total 28.
    
    # Similarly for A22 * B_bot. Total 28.
    
    # Helper to add Strassen terms for a 2x2 * 2x2 block
    # Inputs:
    #   u_base: index offset in U for the 2x2 A block (4 entries)
    #   v_base: index offset in V for the 2x2 B block (4 entries)
    #   w_base: index offset in W for the 2x2 C block (4 entries)
    #   col_start: current column index in U,V,W
    # Returns: new col index (col_start + 7)
    
    def add_strassen_2x2(u_base, v_base, w_base, col_start):
        # Strassen coefficients for 2x2 * 2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # A indices relative to u_base: 00, 01, 10, 11
        # B indices relative to v_base: 00, 01, 10, 11
        # C indices relative to w_base: 00, 01, 10, 11
        
        # Note on index mapping:
        # A is 2x2. U indices: u_base+0 (A00), u_base+1 (A01), u_base+2 (A10), u_base+3 (A11)
        # B is 2x2. V indices: v_base+0 (B00), v_base+1 (B01), v_base+2 (B10), v_base+3 (B11)
        # C is 2x2. W indices: w_base+0 (C00), w_base+1 (C01), w_base+2 (C10), w_base+3 (C11)
        
        # M1
        U[u_base+0, col_start] += 1
        U[u_base+3, col_start] += 1
        V[v_base+0, col_start] += 1
        V[v_base+3, col_start] += 1
        W[w_base+0, col_start] += 1
        W[w_base+3, col_start] += 1
        col_start += 1
        
        # M2
        U[u_base+2, col_start] += 1
        U[u_base+3, col_start] += 1
        V[v_base+0, col_start] += 1
        W[w_base+2, col_start] += 1
        W[w_base+3, col_start] += 1
        col_start += 1
        
        # M3
        U[u_base+0, col_start] += 1
        V[v_base+1, col_start] += 1
        V[v_base+3, col_start] -= 1
        W[w_base+1, col_start] += 1
        W[w_base+3, col_start] += 1
        col_start += 1
        
        # M4
        U[u_base+3, col_start] += 1
        V[v_base+2, col_start] += 1
        V[v_base+0, col_start] -= 1
        W[w_base+0, col_start] += 1
        W[w_base+2, col_start] += 1
        col_start += 1
        
        # M5
        U[u_base+0, col_start] += 1
        U[u_base+1, col_start] += 1
        V[v_base+3, col_start] += 1
        W[w_base+0, col_start] -= 1
        W[w_base+1, col_start] += 1
        col_start += 1
        
        # M6
        U[u_base+2, col_start] += 1
        U[u_base+0, col_start] -= 1
        V[v_base+0, col_start] += 1
        V[v_base+1, col_start] += 1
        W[w_base+3, col_start] += 1
        col_start += 1
        
        # M7
        U[u_base+1, col_start] += 1
        U[u_base+3, col_start] -= 1
        V[v_base+2, col_start] += 1
        V[v_base+3, col_start] += 1
        W[w_base+0, col_start] += 1
        W[w_base+1, col_start] -= 1
        col_start += 1
        
        return col_start

    # --- A21 (2x2) * B_top (2x8) ---
    # A21 is rows 1,2 of A, cols 0,1.
    # U indices for A21:
    # A[1,0] -> 1*4 + 0 = 4
    # A[1,1] -> 1*4 + 1 = 5
    # A[2,0] -> 2*4 + 0 = 8
    # A[2,1] -> 2*4 + 1 = 9
    # So u_base_A21 is not contiguous in a 2x2 block in U (which is 12xR).
    # U is flattened row-major: [A00 A01 A02 A03 A10 A11 A12 A13 A20 A21 A22 A23]
    # A21 (top-left 2x2 of A2) corresponds to A[1,0], A[1,1], A[2,0], A[2,1].
    # Indices: 4, 5, 8, 9.
    
    # We need to handle non-contiguous indices for U.
    # My add_strassen_2x2 assumes contiguous u_base.
    # I will write a more general add_strassen_2x2 that takes explicit index lists.
    
    def add_strassen_2x2_general(u_idx, v_idx, w_idx, col_start):
        # u_idx: list of 4 indices [A00, A01, A10, A11]
        # v_idx: list of 4 indices [B00, B01, B10, B11]
        # w_idx: list of 4 indices [C00, C01, C10, C11]
        
        # M1 = (A00+A11)(B00+B11)
        U[u_idx[0], col_start] += 1
        U[u_idx[3], col_start] += 1
        V[v_idx[0], col_start] += 1
        V[v_idx[3], col_start] += 1
        W[w_idx[0], col_start] += 1
        W[w_idx[3], col_start] += 1
        col_start += 1
        
        # M2 = (A10+A11)B00
        U[u_idx[2], col_start] += 1
        U[u_idx[3], col_start] += 1
        V[v_idx[0], col_start] += 1
        W[w_idx[2], col_start] += 1
        W[w_idx[3], col_start] += 1
        col_start += 1
        
        # M3 = A00(B01-B11)
        U[u_idx[0], col_start] += 1
        V[v_idx[1], col_start] += 1
        V[v_idx[3], col_start] -= 1
        W[w_idx[1], col_start] += 1
        W[w_idx[3], col_start] += 1
        col_start += 1
        
        # M4 = A11(B10-B00)
        U[u_idx[3], col_start] += 1
        V[v_idx[2], col_start] += 1
        V[v_idx[0], col_start] -= 1
        W[w_idx[0], col_start] += 1
        W[w_idx[2], col_start] += 1
        col_start += 1
        
        # M5 = (A00+A01)B11
        U[u_idx[0], col_start] += 1
        U[u_idx[1], col_start] += 1
        V[v_idx[3], col_start] += 1
        W[w_idx[0], col_start] -= 1
        W[w_idx[1], col_start] += 1
        col_start += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[u_idx[2], col_start] += 1
        U[u_idx[0], col_start] -= 1
        V[v_idx[0], col_start] += 1
        V[v_idx[1], col_start] += 1
        W[w_idx[3], col_start] += 1
        col_start += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[u_idx[1], col_start] += 1
        U[u_idx[3], col_start] -= 1
        V[v_idx[2], col_start] += 1
        V[v_idx[3], col_start] += 1
        W[w_idx[0], col_start] += 1
        W[w_idx[1], col_start] -= 1
        col_start += 1
        
        return col_start

    # Process A21 (rows 1,2; cols 0,1) * B_top (rows 0,1; cols 0-7)
    # B_top is split into 4 blocks of 2x2:
    # BT1: rows 0,1; cols 0,1
    # BT2: rows 0,1; cols 2,3
    # BT3: rows 0,1; cols 4,5
    # BT4: rows 0,1; cols 6,7
    
    # U indices for A21:
    # A[1,0]=4, A[1,1]=5, A[2,0]=8, A[2,1]=9
    u_A21 = [4, 5, 8, 9]
    
    # W indices for C part corresponding to A21*B_top
    # C rows are 1,2. C cols are 0-7.
    # C[1, k] index: k*3 + 1
    # C[2, k] index: k*3 + 2
    
    # Block BT1 (cols 0,1)
    # V indices for BT1:
    # B[0,0]=0, B[0,1]=1, B[1,0]=8, B[1,1]=9
    v_BT1 = [0, 1, 8, 9]
    # W indices for C[1,0], C[1,1], C[2,0], C[2,1]
    # C[1,0]=0*3+1=1, C[1,1]=1*3+1=4, C[2,0]=0*3+2=2, C[2,1]=1*3+2=5
    w_C12_c01 = [1, 4, 2, 5]
    col = add_strassen_2x2_general(u_A21, v_BT1, w_C12_c01, col)
    
    # Block BT2 (cols 2,3)
    # V indices for BT2:
    # B[0,2]=2, B[0,3]=3, B[1,2]=10, B[1,3]=11
    v_BT2 = [2, 3, 10, 11]
    # W indices for C[1,2], C[1,3], C[2,2], C[2,3]
    # C[1,2]=2*3+1=7, C[1,3]=3*3+1=10, C[2,2]=2*3+2=8, C[2,3]=3*3+2=11
    w_C12_c23 = [7, 10, 8, 11]
    col = add_strassen_2x2_general(u_A21, v_BT2, w_C12_c23, col)
    
    # Block BT3 (cols 4,5)
    # V indices for BT3:
    # B[0,4]=4, B[0,5]=5, B[1,4]=12, B[1,5]=13
    v_BT3 = [4, 5, 12, 13]
    # W indices for C[1,4], C[1,5], C[2,4], C[2,5]
    # C[1,4]=4*3+1=13, C[1,5]=5*3+1=16, C[2,4]=4*3+2=14, C[2,5]=5*3+2=17
    w_C12_c45 = [13, 16, 14, 17]
    col = add_strassen_2x2_general(u_A21, v_BT3, w_C12_c45, col)
    
    # Block BT4 (cols 6,7)
    # V indices for BT4:
    # B[0,6]=6, B[0,7]=7, B[1,6]=14, B[1,7]=15
    v_BT4 = [6, 7, 14, 15]
    # W indices for C[1,6], C[1,7], C[2,6], C[2,7]
    # C[1,6]=6*3+1=19, C[1,7]=7*3+1=22, C[2,6]=6*3+2=20, C[2,7]=7*3+2=23
    w_C12_c67 = [19, 22, 20, 23]
    col = add_strassen_2x2_general(u_A21, v_BT4, w_C12_c67, col)
    
    # --- A22 (2x2) * B_bot (2x8) ---
    # A22 is rows 1,2 of A, cols 2,3.
    # U indices for A22:
    # A[1,2]=6, A[1,3]=7, A[2,2]=10, A[2,3]=11
    u_A22 = [6, 7, 10, 11]
    
    # B_bot is rows 2,3 of B.
    # Split into 4 blocks of 2x2:
    # BB1: rows 2,3; cols 0,1
    # BB2: rows 2,3; cols 2,3
    # BB3: rows 2,3; cols 4,5
    # BB4: rows 2,3; cols 6,7
    
    # Block BB1 (cols 0,1)
    # V indices for BB1:
    # B[2,0]=16, B[2,1]=17, B[3,0]=24, B[3,1]=25
    v_BB1 = [16, 17, 24, 25]
    # W indices same as C12_c01 because it contributes to C[1,0..1] and C[2,0..1]
    w_C12_c01 = [1, 4, 2, 5]
    col = add_strassen_2x2_general(u_A22, v_BB1, w_C12_c01, col)
    
    # Block BB2 (cols 2,3)
    # V indices for BB2:
    # B[2,2]=18, B[2,3]=19, B[3,2]=26, B[3,3]=27
    v_BB2 = [18, 19, 26, 27]
    w_C12_c23 = [7, 10, 8, 11]
    col = add_strassen_2x2_general(u_A22, v_BB2, w_C12_c23, col)
    
    # Block BB3 (cols 4,5)
    # V indices for BB3:
    # B[2,4]=20, B[2,5]=21, B[3,4]=28, B[3,5]=29
    v_BB3 = [20, 21, 28, 29]
    w_C12_c45 = [13, 16, 14, 17]
    col = add_strassen_2x2_general(u_A22, v_BB3, w_C12_c45, col)
    
    # Block BB4 (cols 6,7)
    # V indices for BB4:
    # B[2,6]=22, B[2,7]=23, B[3,6]=30, B[3,7]=31
    v_BB4 = [22, 23, 30, 31]
    w_C12_c67 = [19, 22, 20, 23]
    col = add_strassen_2x2_general(u_A22, v_BB4, w_C12_c67, col)
    
    # Verify col is 88
    assert col == 88, f"Expected col 88, got {col}"
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
