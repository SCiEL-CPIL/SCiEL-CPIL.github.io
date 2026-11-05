import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition for 2x4x7
    # Split A into two 2x2 blocks: A1, A2
    # Split B into two 2x7 blocks: B1, B2
    # C = A1 B1 + A2 B2
    
    # Helper to get Strassen 2x2x2 decomposition
    # Returns U (4x7), V (4x7), W (4x7) for 2x2x2 multiplication
    # Indices for A: 0,1 (row 0), 2,3 (row 1)
    # Indices for B: 0,1 (col 0), 2,3 (col 1)
    # Indices for C: 0,1 (row 0), 2,3 (row 1)
    
    def strassen_222():
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        W[1, 0] = -1; W[2, 0] = -1 # C11=M1+M4-M5+M7, C22=M1-M2+M3+M6
        # Wait, W is for C.
        # C11 gets M1, C12 gets 0, C21 gets 0, C22 gets M1
        # So W[0,0]=1, W[3,0]=1.
        
        # Let's be careful with W signs.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    # Helper for 2x2x1
    def direct_221():
        # C1 = A11 B11 + A21 B21
        # C2 = A12 B12 + A22 B22
        U = np.zeros((4, 4))
        V = np.zeros((2, 4)) # B is 2x1, so 2 entries
        W = np.zeros((2, 4)) # C is 2x1, so 2 entries
        
        # T1: A11 B11
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # T2: A21 B21
        U[2, 1] = 1
        V[1, 1] = 1
        W[0, 1] = 1
        
        # T3: A12 B12
        U[1, 2] = 1
        V[0, 2] = 1
        W[1, 2] = 1
        
        # T4: A22 B22
        U[3, 3] = 1
        V[1, 3] = 1
        W[1, 3] = 1
        
        return U, V, W

    # Build 2x2x7 decomposition
    # 3 blocks of 2x2x2, 1 block of 2x2x1
    U_227 = []
    V_227 = []
    W_227 = []
    
    # Block 1 (cols 0,1)
    U1, V1, W1 = strassen_222()
    U_227.append(U1)
    V_227.append(V1)
    W_227.append(W1)
    
    # Block 2 (cols 2,3)
    U2, V2, W2 = strassen_222()
    # Shift V indices by 2
    V2_shifted = np.zeros((4, 7))
    V2_shifted[:, 2:] = V2
    # Shift W indices by 2
    W2_shifted = np.zeros((4, 7))
    W2_shifted[:, 2:] = W2
    U_227.append(U2)
    V_227.append(V2_shifted)
    W_227.append(W2_shifted)
    
    # Block 3 (cols 4,5)
    U3, V3, W3 = strassen_222()
    V3_shifted = np.zeros((4, 7))
    V3_shifted[:, 4:] = V3
    W3_shifted = np.zeros((4, 7))
    W3_shifted[:, 4:] = W3
    U_227.append(U3)
    V_227.append(V3_shifted)
    W_227.append(W3_shifted)
    
    # Block 4 (col 6)
    U4, V4, W4 = direct_221()
    V4_shifted = np.zeros((2, 4)) # V4 is 2x4
    # V4 is for 2x1 B. We need to map to 2x7 B.
    # V4[:, k] corresponds to B[?, 6]
    V4_full = np.zeros((4, 4))
    V4_full[:, 0] = V4[:, 0]
    V4_full[:, 1] = V4[:, 1]
    V4_full[:, 2] = V4[:, 2]
    V4_full[:, 3] = V4[:, 3]
    # Wait, V4 is 2x4. The columns are the 4 terms.
    # The rows are the 2 entries of B (row 0 and row 1).
    # We need to place these into the 4x7 V matrix for 2x2x7.
    # The B part of 2x2x7 is 2x7. Flattened 4x? No, 2x7 is 14 entries?
    # No, V is (m*p, rank). m=2, p=7. So V is 14xRank.
    # My strassen_222 V was 4x7. That's for 2x2 B (4 entries).
    # So for 2x2x7, V should be 14xRank.
    
    # Let's restart the V/W shape logic.
    # A is 2x2 (4 entries). U is 4xRank.
    # B is 2x7 (14 entries). V is 14xRank.
    # C is 2x7 (14 entries). W is 14xRank.
    
    # strassen_222 returns U(4x7), V(4x7), W(4x7).
    # V(4x7) corresponds to B(2x2).
    # To map to B(2x7), we need to expand V.
    
    def expand_V(V_22, start_col):
        # V_22 is 4xRank. Rows 0,1 are B[0,0], B[0,1]. Rows 2,3 are B[1,0], B[1,1].
        # We want V_27 to be 14xRank.
        # Rows 0..6 are B[0,0]..B[0,6]. Rows 7..13 are B[1,0]..B[1,6].
        # V_22[:, k] has values for B[0,0], B[0,1], B[1,0], B[1,1].
        # We place them at B[0, start_col], B[0, start_col+1], B[1, start_col], B[1, start_col+1].
        V_27 = np.zeros((14, V_22.shape[1]))
        V_27[start_col, :] = V_22[0, :]
        V_27[start_col+1, :] = V_22[1, :]
        V_27[7+start_col, :] = V_22[2, :]
        V_27[7+start_col+1, :] = V_22[3, :]
        return V_27

    def expand_W(W_22, start_col):
        # W_22 is 4xRank. Rows 0,1 are C[0,0], C[0,1]. Rows 2,3 are C[1,0], C[1,1].
        # We want W_27 to be 14xRank.
        W_27 = np.zeros((14, W_22.shape[1]))
        W_27[start_col, :] = W_22[0, :]
        W_27[start_col+1, :] = W_22[1, :]
        W_27[7+start_col, :] = W_22[2, :]
        W_27[7+start_col+1, :] = W_22[3, :]
        return W_27

    # Re-build 2x2x7
    U_227 = []
    V_227 = []
    W_227 = []
    
    U1, V1, W1 = strassen_222()
    U_227.append(U1)
    V_227.append(expand_V(V1, 0))
    W_227.append(expand_W(W1, 0))
    
    U2, V2, W2 = strassen_222()
    U_227.append(U2)
    V_227.append(expand_V(V2, 2))
    W_227.append(expand_W(W2, 2))
    
    U3, V3, W3 = strassen_222()
    U_227.append(U3)
    V_227.append(expand_V(V3, 4))
    W_227.append(expand_W(W3, 4))
    
    U4, V4, W4 = direct_221()
    # V4 is 2x4. Rows 0,1 are B[0,0], B[1,0].
    # We need to expand to 14x4.
    V4_27 = np.zeros((14, 4))
    V4_27[6, :] = V4[0, :] # B[0,6]
    V4_27[13, :] = V4[1, :] # B[1,6]
    U_227.append(U4)
    V_227.append(V4_27)
    
    # W4 is 2x4. Rows 0,1 are C[0,0], C[1,0].
    W4_27 = np.zeros((14, 4))
    W4_27[6, :] = W4[0, :] # C[0,6]
    W4_27[13, :] = W4[1, :] # C[1,6]
    W_227.append(W4_27)
    
    U_227 = np.hstack(U_227)
    V_227 = np.hstack(V_227)
    W_227 = np.hstack(W_227)
    
    # Now we have 2x2x7 decomposition.
    # We need 2x4x7.
    # A is 2x4. Split into A1 (2x2), A2 (2x2).
    # B is 4x7. Split into B1 (2x7), B2 (2x7).
    # C is 2x7. C = C1 + C2.
    
    # U_227 is 4x25. Corresponds to A1.
    # We need U for A (8 entries).
    # U_full will be 8x50.
    # First 25 columns: U_227 for A1. Rows 0-3 are A1. Rows 4-7 are 0.
    # Last 25 columns: U_227 for A2. Rows 0-3 are 0. Rows 4-7 are U_227.
    
    U_full = np.zeros((8, 50))
    U_full[0:4, 0:25] = U_227
    U_full[4:8, 25:50] = U_227
    
    # V_227 is 14x25. Corresponds to B1 (2x7).
    # We need V for B (4x7 = 28 entries).
    # V_full will be 28x50.
    # First 25 columns: V_227 for B1. Rows 0-13 are B1. Rows 14-27 are 0.
    # Last 25 columns: V_227 for B2. Rows 0-13 are 0. Rows 14-27 are V_227.
    
    V_full = np.zeros((28, 50))
    V_full[0:14, 0:25] = V_227
    V_full[14:28, 25:50] = V_227
    
    # W_227 is 14x25. Corresponds to C (2x7).
    # C1 and C2 are both 2x7.
    # W_full will be 14x50.
    # First 25 columns: W_227.
    # Last 25 columns: W_227.
    
    W_full = np.hstack([W_227, W_227])
    
    return {"U": U_full, "V": V_full, "W": W_full}

if __name__ == "__main__":
    res = construct()
    print(res["U"].shape)
