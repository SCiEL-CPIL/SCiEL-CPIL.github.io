import numpy as np

def construct():
    # We will build the U, V, W matrices for the final 2x5x6 problem.
    # Dimensions: n=2, m=5, p=6.
    # U: (2*5, rank) = (10, rank)
    # V: (5*6, rank) = (30, rank)
    # W: (6*2, rank) = (12, rank)
    
    # The decomposition is:
    # 1. Kronecker Product of T1 (2x5x2) and T2 (1x1x3).
    # T1 is composed of:
    #   - T1a: 2x2x2 (Strassen, rank 7)
    #   - T1b: 2x2x2 (Strassen, rank 7)
    #   - T1c: 2x1x2 (Outer, rank 4)
    # T2 is 1x1x3 (rank 3).
    
    # Total rank = (7 + 7 + 4) * 3 = 54.
    
    rank = 54
    U = np.zeros((10, rank))
    V = np.zeros((30, rank))
    W = np.zeros((12, rank))
    
    # Helper to add a decomposition block
    # block_U: (n_block*m_block, r_block)
    # block_V: (m_block*p_block, r_block)
    # block_W: (p_block*n_block, r_block)
    # offsets: u_off, v_off, w_off
    # scale: not needed as entries are integers
    
    def add_block(bU, bV, bW, u_off, v_off, w_off, col_start):
        r_b = bU.shape[1]
        U[u_off:u_off+bU.shape[0], col_start:col_start+r_b] = bU
        V[v_off:v_off+bV.shape[0], col_start:col_start+r_b] = bV
        W[w_off:w_off+bW.shape[0], col_start:col_start+r_b] = bW
        return col_start + r_b
    
    # Define Strassen 2x2x2 block
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
    def get_strassen_222():
        r = 7
        bU = np.zeros((4, r))
        bV = np.zeros((4, r))
        bW = np.zeros((4, r))
        
        # M1 = (A11+A22)(B11+B22)
        bU[0, 0] = 1; bU[3, 0] = 1
        bV[0, 0] = 1; bV[3, 0] = 1
        bW[0, 0] = 1; bW[3, 0] = 1
        
        # M2 = (A21+A22)B11
        bU[2, 1] = 1; bU[3, 1] = 1
        bV[0, 1] = 1
        bW[2, 1] = 1; bW[3, 1] = 1 # C21 += M2, C22 -= M2? 
        # Formula: C11 = M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
        # Let's re-verify W coefficients for Strassen.
        
        # C11 (idx 0): M1(1) + M4(1) - M5(1) + M7(1)
        # C12 (idx 1): M3(1) + M5(1)
        # C21 (idx 2): M2(1) + M4(1)
        # C22 (idx 3): M1(1) - M2(1) + M3(1) + M6(1)
        
        # M1
        bW[0, 0] = 1; bW[3, 0] = 1
        # M2
        bW[2, 1] = 1; bW[3, 1] = -1
        # M3
        bW[1, 2] = 1; bW[3, 2] = 1
        # M4
        bW[0, 3] = 1; bW[2, 3] = 1
        # M5
        bW[0, 4] = -1; bW[1, 4] = 1
        # M6
        bW[3, 5] = 1
        # M7
        bW[0, 6] = 1
        
        # U and V for Strassen
        # M1: A11+A22, B11+B22
        bU[0, 0] = 1; bU[3, 0] = 1
        bV[0, 0] = 1; bV[3, 0] = 1
        
        # M2: A21+A22, B11
        bU[2, 1] = 1; bU[3, 1] = 1
        bV[0, 1] = 1
        
        # M3: A11, B12-B22
        bU[0, 2] = 1
        bV[1, 2] = 1; bV[3, 2] = -1
        
        # M4: A22, B21-B11
        bU[3, 3] = 1
        bV[2, 3] = 1; bV[0, 3] = -1
        
        # M5: A11+A12, B22
        bU[0, 4] = 1; bU[1, 4] = 1
        bV[3, 4] = 1
        
        # M6: A21-A11, B11+B12
        bU[2, 5] = 1; bU[0, 5] = -1
        bV[0, 5] = 1; bV[1, 5] = 1
        
        # M7: A12-A22, B21+B22
        bU[1, 6] = 1; bU[3, 6] = -1
        bV[2, 6] = 1; bV[3, 6] = 1
        
        return bU, bV, bW

    def get_outer_212():
        # 2x1x2
        # A (2x1), B (1x2), C (2x2)
        # C11 = A1 B1
        # C12 = A1 B2
        # C21 = A2 B1
        # C22 = A2 B2
        r = 4
        bU = np.zeros((2, r)) # n*m = 2*1 = 2
        bV = np.zeros((2, r)) # m*p = 1*2 = 2
        bW = np.zeros((4, r)) # p*n = 2*2 = 4
        
        bU[0, 0] = 1; bV[0, 0] = 1; bW[0, 0] = 1
        bU[0, 1] = 1; bV[1, 1] = 1; bW[1, 1] = 1
        bU[1, 2] = 1; bV[0, 2] = 1; bW[2, 2] = 1
        bU[1, 3] = 1; bV[1, 3] = 1; bW[3, 3] = 1
        
        return bU, bV, bW

    def get_kron_113():
        # 1x1x3
        # A (1x1), B (1x3), C (1x3)
        # Ck = A1 Bk
        r = 3
        bU = np.zeros((1, r))
        bV = np.zeros((3, r))
        bW = np.zeros((3, r))
        
        for k in range(3):
            bU[0, k] = 1
            bV[k, k] = 1
            bW[k, k] = 1
            
        return bU, bV, bW

    # Get blocks
    sU, sV, sW = get_strassen_222()
    oU, oV, oW = get_outer_212()
    kU, kV, kW = get_kron_113()
    
    # Construct T1 (2x5x2)
    # T1a: 2x2x2 (Strassen) at m-offset 0
    # T1b: 2x2x2 (Strassen) at m-offset 2
    # T1c: 2x1x2 (Outer) at m-offset 4
    
    # We will build T1 first, then Kronecker with T2.
    # Actually, it's easier to build the final U, V, W directly by iterating over the Kronecker structure.
    
    # T1 Rank = 18. T2 Rank = 3. Total 54.
    # For each of the 3 "columns" of T2 (k=0,1,2):
    #   We add the T1 block scaled by T2 entries.
    #   Since T2 entries are 1, we just copy T1 block.
    #   But we need to map the indices correctly.
    
    # T1 structure:
    # U1: (10, 18)
    # V1: (10, 18)  # m*p for 2x5x2 is 5*2=10
    # W1: (4, 18)   # p*n for 2x5x2 is 2*2=4
    
    U1 = np.zeros((10, 18))
    V1 = np.zeros((10, 18))
    W1 = np.zeros((4, 18))
    
    col = 0
    # Block 1: Strassen 2x2x2
    # A rows 0,1 (m=0,1). B rows 0,1 (m=0,1). C rows 0,1 (p=0,1).
    U1[0:2, col:col+7] = sU
    V1[0:2, col:col+7] = sV
    W1[0:2, col:col+7] = sW
    col += 7
    
    # Block 2: Strassen 2x2x2
    # A rows 2,3 (m=2,3). B rows 2,3 (m=2,3). C rows 0,1 (p=0,1).
    U1[2:4, col:col+7] = sU
    V1[2:4, col:col+7] = sV
    W1[0:2, col:col+7] = sW
    col += 7
    
    # Block 3: Outer 2x1x2
    # A row 4 (m=4). B row 4 (m=4). C rows 0,1 (p=0,1).
    # oU is 2x4. But here A is 2x1, so only 1 row of A?
    # Wait, A is 2x5. Row 0 is m=0,1,2,3,4? No.
    # A is 2x5.
    # Row 0 of A: A[0,0], A[0,1], A[0,2], A[0,3], A[0,4].
    # Row 1 of A: A[1,0], A[1,1], A[1,2], A[1,3], A[1,4].
    # My U1 indexing:
    # U1 is (10, 18).
    # Rows 0,1 correspond to A[0,0], A[1,0]?
    # Standard flattening: U[i*m + j].
    # i=0, j=0 -> 0. i=0, j=1 -> 1. i=1, j=0 -> 2. i=1, j=1 -> 3.
    # So for 2x2 block (j=0,1):
    # U rows: 0 (A00), 1 (A01), 2 (A10), 3 (A11).
    # My sU is 4x7.
    # sU[0] is A11 (in local 2x2). Maps to A00?
    # In Strassen, indices are 1-based (1,1), (1,2), (2,1), (2,2).
    # Local (1,1) -> Global (0,0) -> U row 0.
    # Local (1,2) -> Global (0,1) -> U row 1.
    # Local (2,1) -> Global (1,0) -> U row 2.
    # Local (2,2) -> Global (1,1) -> U row 3.
    
    # So for Block 1 (m=0,1):
    # U1[0:4, 0:7] = sU.
    # V1[0:4, 0:7] = sV. (B local (1,1)->0, (1,2)->1, (2,1)->2, (2,2)->3).
    # W1[0:4, 0:7] = sW. (C local (1,1)->0, (1,2)->1, (2,1)->2, (2,2)->3).
    
    # Let's re-assign U1, V1, W1 correctly.
    U1 = np.zeros((10, 18))
    V1 = np.zeros((10, 18))
    W1 = np.zeros((4, 18))
    
    # Block 1: m=0,1
    U1[0:4, 0:7] = sU
    V1[0:4, 0:7] = sV
    W1[0:4, 0:7] = sW
    
    # Block 2: m=2,3
    # A rows: A02(4), A03(5), A12(6), A13(7).
    # B rows: B20(4), B21(5), B30(6), B31(7).
    # C rows: same as Block 1.
    U1[4:8, 7:14] = sU
    V1[4:8, 7:14] = sV
    W1[0:4, 7:14] = sW
    
    # Block 3: m=4
    # A rows: A04(8), A14(9).
    # B rows: B40(8), B41(9).
    # C rows: same.
    # oU is 2x4. Rows 0,1 correspond to A04, A14?
    # In get_outer_212, U is 2x4.
    # Row 0: A1 (local). Row 1: A2 (local).
    # So U1[8, 14:18] = oU[0]
    #    U1[9, 14:18] = oU[1]
    # V1[8, 14:18] = oV[0]
    #    V1[9, 14:18] = oV[1]
    # W1[0:4, 14:18] = oW
    
    U1[8:10, 14:18] = oU
    V1[8:10, 14:18] = oV
    W1[0:4, 14:18] = oW
    
    # Now Kronecker with T2 (1x1x3)
    # T2 has rank 3.
    # For each k in 0,1,2:
    #   We add T1 block.
    #   U_final: rows 0..9.
    #   V_final: rows 0..29.
    #   W_final: rows 0..11.
    
    # V_final structure:
    # B is 5x6.
    # Flattened: V[j*6 + k].
    # For a fixed k (0,1,2), the B part is B[:, k].
    # This is a 5x1 vector.
    # In T1, V1 is 10x18.
    # V1 rows correspond to B[:, 0] and B[:, 1] (since p=2 in T1).
    # Wait, T1 is 2x5x2. p=2.
    # V1 rows: 0..4 for B[:,0], 5..9 for B[:,1].
    # In Final V (5x6), for a fixed k in 0..5:
    #   If k=0, we use B[:,0]. This maps to V1 rows 0..4.
    #   If k=1, we use B[:,1]. This maps to V1 rows 5..9.
    #   If k=2, we use B[:,2]. This is a new column in B, not in T1?
    #   No, T2 is 1x1x3.
    #   The Kronecker product T1 \otimes T2.
    #   T1: 2x5x2. T2: 1x1x3.
    #   Result: 2x5x6.
    #   The "p" dimension of T1 (size 2) is multiplied by "p" dimension of T2 (size 3).
    #   So the 6 columns of B_final are grouped into 3 groups of 2?
    #   Or 2 groups of 3?
    #   p_final = p1 * p2 = 2 * 3 = 6.
    #   Index k_final = k1 * p2 + k2? Or k2 * p1 + k1?
    #   Usually Kronecker product of matrices:
    #   If A is n1 x m1, B is m1 x p1.
    #   C is n1 x p1.
    #   Kronecker with A2 (n2 x m2), B2 (m2 x p2).
    #   C_final = C1 \otimes C2? No.
    #   The multiplication (A1 A2) (B1 B2) = (A1 B1) (A2 B2)? No.
    #   (A1 \otimes A2) (B1 \otimes B2) = (A1 B1) \otimes (A2 B2).
    #   Here A is 2x5. B is 5x6.
    #   We wrote A = A1 \otimes A2?
    #   A1 (2x5), A2 (1x1)? No.
    #   We used T1 (2x5x2) and T2 (1x1x3).
    #   This implies A = A1 \otimes A2 where A1 is 2x5, A2 is 1x1.
    #   B = B1 \otimes B2 where B1 is 5x2, B2 is 1x3.
    #   Then C = C1 \otimes C2 where C1 is 2x2, C2 is 1x3.
    #   C1 is 2x2. C2 is 1x3.
    #   C_final is 2x6.
    #   Index in C_final: (i, k).
    #   k = k1 * 3 + k2?
    #   B1 is 5x2. B2 is 1x3.
    #   B_final is 5x6.
    #   B_final[j, k1*3 + k2] = B1[j, k1] * B2[0, k2].
    #   Since B2 is 1x3, B2[0, k2] is just a scalar.
    #   So B_final[:, k1*3 + k2] = B1[:, k1] * B2[0, k2].
    
    # So for each k2 in 0,1,2:
    #   We use the T1 decomposition.
    #   The V part of T1 corresponds to B1 (5x2).
    #   In V_final (30x54):
    #   Rows for B_final[:, k1*3 + k2] are at index j*6 + (k1*3 + k2).
    #   We need to map V1 rows (which are for B1[:, k1]) to V_final rows.
    #   V1 has 10 rows.
    #   Rows 0-4 correspond to B1[:, 0].
    #   Rows 5-9 correspond to B1[:, 1].
    
    #   For a fixed k2:
    #   For k1=0:
    #     V_final[j*6 + 0*3 + k2, col] = V1[j, col] * B2[0, k2]
    #     Since B2[0, k2] = 1, V_final[j*6 + k2, col] = V1[j, col].
    #   For k1=1:
    #     V_final[j*6 + 1*3 + k2, col] = V1[j+5, col].
    
    #   W_final:
    #   C_final[i, k1*3 + k2] = C1[i, k1] * C2[0, k2].
    #   W_final[k1*3 + k2, i, col] = W1[k1, i, col].
    #   W1 is 4x18. Rows are (k1, i).
    #   Row index in W1: k1*2 + i.
    #   Row index in W_final: (k1*3 + k2)*2 + i.
    
    #   U_final:
    #   U_final[i*5 + j, col] = U1[i*5 + j, col].
    #   Same for all k2.
    
    U_final = np.zeros((10, 54))
    V_final = np.zeros((30, 54))
    W_final = np.zeros((12, 54))
    
    for k2 in range(3):
        col_start = k2 * 18
        # U
        U_final[:, col_start:col_start+18] = U1
        
        # V
        # k1=0
        for j in range(5):
            V_final[j*6 + k2, col_start:col_start+18] = V1[j, :]
        # k1=1
        for j in range(5):
            V_final[j*6 + 3 + k2, col_start:col_start+18] = V1[j+5, :]
            
        # W
        # k1=0
        for i in range(2):
            W_final[k2*2 + i, col_start:col_start+18] = W1[i, :]
        # k1=1
        for i in range(2):
            W_final[(3+k2)*2 + i, col_start:col_start+18] = W1[i+2, :]
            
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
