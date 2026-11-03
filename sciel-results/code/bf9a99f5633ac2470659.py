import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Strassen decomposition for 2x2 matrix multiplication
    # T = sum_{r=1}^7 U_r * V_r * W_r
    # U, V, W are 4x7 matrices
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # r=0
    U_s[0,0]=1; U_s[3,0]=1
    V_s[0,0]=1; V_s[3,0]=1
    W_s[0,0]=1; W_s[3,0]=1
    
    # r=1
    U_s[2,1]=1; U_s[3,1]=1
    V_s[0,1]=1
    W_s[2,1]=1; W_s[3,1]=-1
    
    # r=2
    U_s[0,2]=1
    V_s[1,2]=1; V_s[3,2]=-1
    W_s[1,2]=1; W_s[3,2]=1
    
    # r=3
    U_s[3,3]=1
    V_s[2,3]=1; V_s[0,3]=-1
    W_s[0,3]=1; W_s[2,3]=1
    
    # r=4
    U_s[0,4]=1; U_s[1,4]=1
    V_s[3,4]=1
    W_s[0,4]=-1; W_s[1,4]=1
    
    # r=5
    U_s[2,5]=1; U_s[0,5]=-1
    V_s[0,5]=1; V_s[1,5]=1
    W_s[3,5]=1
    
    # r=6
    U_s[1,6]=1; U_s[3,6]=-1
    V_s[2,6]=1; V_s[3,6]=1
    W_s[0,6]=1

    # Construct 2x4 * 4x8 decomposition
    # Split A (2x4) into A0, A1 (2x2)
    # Split B (4x8) into B0, B1 (2x8)
    # C = A0 B0 + A1 B1
    # Each term is 2x2 * 2x8
    # Split B0 (2x8) into B00, B01 (2x4)
    # A0 B0 = A0 B00 + A0 B01
    # Each term is 2x2 * 2x4
    # Rank of 2x2 * 2x4 is 14 (2 * rank(2x2*2x2))
    # Total rank = 2 * (2 * 14) = 56
    
    rank = 56
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Mapping for U (2x4)
    # A0 is top 2x2, A1 is bottom 2x2? No, A is 2x4.
    # A = [A_col0 A_col1] where A_col0, A_col1 are 2x2.
    # Let's call them A_L and A_R.
    # U indices: i*m + j. i in 0..1, j in 0..3.
    # A_L corresponds to j=0,1. A_R corresponds to j=2,3.
    
    # Mapping for V (4x8)
    # B = [B_row0; B_row1] where B_row0, B_row1 are 2x8.
    # Let's call them B_T and B_B.
    # V indices: j*p + k. j in 0..3, k in 0..7.
    # B_T corresponds to j=0,1. B_B corresponds to j=2,3.
    
    # Mapping for W (8x2)
    # C is 2x8.
    # W indices: k*n + i. k in 0..7, i in 0..1.
    
    # We have 4 blocks of 2x2 * 2x4 multiplication.
    # Block 1: A_L * B_T0 (B_T first 4 cols)
    # Block 2: A_L * B_T1 (B_T last 4 cols)
    # Block 3: A_R * B_B0 (B_B first 4 cols)
    # Block 4: A_R * B_B1 (B_B last 4 cols)
    
    # Each block has rank 14.
    # Total rank 56.
    
    def add_block(U_block, V_block, W_block, u_offset, v_offset, w_offset, r_offset):
        # U_block is 4x14, V_block is 8x14, W_block is 8x14
        # u_offset: start row in U
        # v_offset: start row in V
        # w_offset: start row in W
        # r_offset: start col in U,V,W
        U[u_offset:u_offset+4, r_offset:r_offset+14] = U_block
        V[v_offset:v_offset+8, r_offset:r_offset+14] = V_block
        W[w_offset:w_offset+8, r_offset:r_offset+14] = W_block

    # Construct U_block, V_block, W_block for 2x2 * 2x4
    # 2x2 * 2x4 = 2 * (2x2 * 2x2)
    # U_224 is 4x14
    # V_224 is 8x14
    # W_224 is 8x14
    
    U_224 = np.zeros((4, 14))
    V_224 = np.zeros((8, 14))
    W_224 = np.zeros((8, 14))
    
    # First 2x2 * 2x2 (cols 0-3 of B)
    U_224[:, 0:7] = U_s
    # V for B0 (2x2)
    # V indices for B0: j=0,1 and k=0,1,2,3? No.
    # B is 2x4. B0 is 2x2 (cols 0,1). B1 is 2x2 (cols 2,3).
    # V_224[:, 0:7] corresponds to B0.
    # V_s is 4x7. V_s rows are j*p+k? No, V_s is for 2x2*2x2.
    # V_s indices: j*2+k.
    # In 2x2*2x4, V is 2x4. Index j*4+k.
    # B0 is j=0,1 and k=0,1.
    # So V_224[0:4, 0:7] = V_s.
    V_224[0:4, 0:7] = V_s
    # W for C0 (2x2). C0 is cols 0,1 of C.
    # W_s indices: k*2+i.
    # W_224[0:4, 0:7] = W_s.
    W_224[0:4, 0:7] = W_s
    
    # Second 2x2 * 2x2 (cols 2-3 of B)
    U_224[:, 7:14] = U_s
    # V for B1 (2x2). B1 is j=0,1 and k=2,3.
    # V_224[4:8, 7:14] = V_s.
    V_224[4:8, 7:14] = V_s
    # W for C1 (2x2). C1 is cols 2,3 of C.
    # W_224[4:8, 7:14] = W_s.
    W_224[4:8, 7:14] = W_s

    # Now assemble the 56 rank decomposition
    
    # Block 1: A_L * B_T0
    # A_L is rows 0,1 of A. U offset 0.
    # B_T0 is rows 0,1 of B and cols 0,3 of B_T?
    # B_T is 2x8. B_T0 is cols 0-3.
    # V offset for B_T0: rows 0,1 of B.
    # W offset for C0: cols 0-3 of C.
    add_block(U_224, V_224, W_224, 0, 0, 0, 0)
    
    # Block 2: A_L * B_T1
    # A_L is rows 0,1 of A. U offset 0.
    # B_T1 is rows 0,1 of B and cols 4-7 of B_T.
    # V offset for B_T1: rows 0,1 of B, but V_224 expects 8 rows.
    # Wait, V_224 is 8x14.
    # In Block 2, we are multiplying A_L (2x2) by B_T1 (2x4).
    # B_T1 corresponds to B rows 0,1 and cols 4-7.
    # In the global V matrix (32x56), the rows for B_T1 are 4,5,6,7?
    # B is 4x8.
    # B_T is rows 0,1. B_B is rows 2,3.
    # B_T0 is cols 0-3. B_T1 is cols 4-7.
    # V indices: j*8+k.
    # B_T0: j=0,1, k=0..3. Indices 0..7.
    # B_T1: j=0,1, k=4..7. Indices 8..15.
    # So V offset for Block 2 is 8.
    # W offset for C1 (cols 4-7 of C).
    # W indices: k*2+i.
    # C0: k=0..3. Indices 0..7.
    # C1: k=4..7. Indices 8..15.
    # So W offset for Block 2 is 8.
    add_block(U_224, V_224, W_224, 0, 8, 8, 14)
    
    # Block 3: A_R * B_B0
    # A_R is rows 0,1 of A? No, A is 2x4.
    # A_L is cols 0,1. A_R is cols 2,3.
    # U indices: i*4+j.
    # A_L: j=0,1. Indices 0,1,4,5.
    # A_R: j=2,3. Indices 2,3,6,7.
    # So U offset for A_R is not a simple contiguous block of 4 rows.
    # U_224 is 4x14.
    # We need to place U_224 into U such that:
    # U_224[0,:] -> U[0,:] (i=0, j=0)
    # U_224[1,:] -> U[1,:] (i=0, j=1)
    # U_224[2,:] -> U[4,:] (i=1, j=0)
    # U_224[3,:] -> U[5,:] (i=1, j=1)
    # This is for A_L.
    
    # For A_R:
    # U_224[0,:] -> U[2,:] (i=0, j=2)
    # U_224[1,:] -> U[3,:] (i=0, j=3)
    # U_224[2,:] -> U[6,:] (i=1, j=2)
    # U_224[3,:] -> U[7,:] (i=1, j=3)
    
    # Let's re-verify U mapping.
    # U shape (8, 56).
    # U[i*4+j, r].
    
    # Block 3: A_R * B_B0
    # U offset: map U_224 rows to U rows [2,3,6,7].
    # V offset: B_B0. B_B is rows 2,3. B_B0 is cols 0-3.
    # V indices: j*8+k.
    # B_B0: j=2,3, k=0..3. Indices 16..23.
    # So V offset is 16.
    # W offset: C0 (cols 0-3). W offset 0.
    
    # We cannot use add_block directly for U because of the non-contiguous rows.
    # We will fill U manually for blocks 3 and 4.
    
    # Block 3
    r_off = 28
    U[2, r_off:r_off+14] = U_224[0, :]
    U[3, r_off:r_off+14] = U_224[1, :]
    U[6, r_off:r_off+14] = U_224[2, :]
    U[7, r_off:r_off+14] = U_224[3, :]
    V[16:24, r_off:r_off+14] = V_224
    W[0:8, r_off:r_off+14] = W_224
    
    # Block 4: A_R * B_B1
    # U offset: same as Block 3.
    # V offset: B_B1. B_B is rows 2,3. B_B1 is cols 4-7.
    # V indices: j=2,3, k=4..7. Indices 24..31.
    # So V offset is 24.
    # W offset: C1 (cols 4-7). W offset 8.
    
    r_off = 42
    U[2, r_off:r_off+14] = U_224[0, :]
    U[3, r_off:r_off+14] = U_224[1, :]
    U[6, r_off:r_off+14] = U_224[2, :]
    U[7, r_off:r_off+14] = U_224[3, :]
    V[24:32, r_off:r_off+14] = V_224
    W[8:16, r_off:r_off+14] = W_224

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
