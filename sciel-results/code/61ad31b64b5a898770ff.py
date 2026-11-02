import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def get_strassen_2x2_factors():
    # Returns U, V, W for 2x2x2 multiplication with rank 7
    # U: (4, 7) for A (2x2)
    # V: (4, 7) for B (2x2)
    # W: (4, 7) for C (2x2)
    
    # Indices for 2x2 matrix A:
    # 0: A00, 1: A01, 2: A10, 3: A11
    # Same for B, C.
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1; W[1, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1; W[0, 6] = 1
    
    return U, V, W

def get_strassen_4x4_factors():
    # Use Kronecker product of 2x2 factors
    U2, V2, W2 = get_strassen_2x2_factors()
    
    # The 4x4 matrix is composed of 2x2 blocks.
    # A_4x4 = [A11 A12; A21 A22] where each is 2x2.
    # The tensor for 4x4x4 is the Kronecker product of the 2x2x2 tensor with itself.
    # U_4x4 = U2 (kron) U2
    # V_4x4 = V2 (kron) V2
    # W_4x4 = W2 (kron) W2
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    return U4, V4, W4

def get_schoolbook_4x4x1_factors():
    # Rank 16
    # A: 4x4, B: 4x1, C: 4x1
    U = np.zeros((16, 16))
    V = np.zeros((4, 16))
    W = np.zeros((4, 16))
    
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            U[i * 4 + j, idx] = 1
            V[j, idx] = 1
            W[i, idx] = 1
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS # 4, 4, 9
    
    # Get 4x4x4 factors
    U4, V4, W4 = get_strassen_4x4_factors()
    # U4: (16, 49), V4: (16, 49), W4: (16, 49)
    
    # Get 4x4x1 factors
    U1, V1, W1 = get_schoolbook_4x4x1_factors()
    # U1: (16, 16), V1: (4, 16), W1: (4, 16)
    
    rank = 49 + 49 + 16
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: C[0:4, 0:4] = A * B[0:4, 0:4]
    # A is 4x4. B part is 4x4. C part is 4x4.
    # U columns 0:49
    U[:, 0:49] = U4
    # V rows 0:16 (B cols 0-3)
    V[0:16, 0:49] = V4
    # W rows for C[0:4, 0:4].
    # C is 4x9. C[0:4, 0:4] means rows 0-3, cols 0-3.
    # W convention: W[k*n + i, r].
    # For C[i, k], row is k*n + i.
    # i in 0..3, k in 0..3.
    # Rows: 0*4+0=0, 1*4+0=4, ..., 3*4+3=15.
    # So W[0:16, 0:49] = W4?
    # Wait, W4 from Kronecker is for 4x4 C.
    # The indices in W4 correspond to C flattened.
    # If W4 is constructed via kron, does it match the k*n+i ordering?
    # U2, V2, W2 are for 2x2.
    # W2 row index corresponds to C 2x2 flattened.
    # If C2 is 2x2, C2[i,j]. Row index in W2?
    # In get_strassen_2x2_factors, W is (4, 7).
    # W[0, t] corresponds to C[0,0]?
    # Let's check the mapping.
    # Standard Kronecker product preserves the structure if the ordering is consistent.
    # If we assume row-major flattening for the 2x2 case:
    # C[0,0]->0, C[0,1]->1, C[1,0]->2, C[1,1]->3.
    # Then for 4x4, row-major:
    # C[0,0]->0, C[0,1]->1, C[0,2]->2, C[0,3]->3
    # C[1,0]->4, ...
    # But the task requires W[k*n + i, r].
    # This is column-major flattening of C.
    # C[0,0]->0, C[1,0]->4, C[2,0]->8, C[3,0]->12
    # C[0,1]->1, ...
    # So we need to permute the rows of W4.
    
    # Let's verify the W4 ordering.
    # In get_strassen_2x2_factors, we set W[0, 0]=1 for C[0,0].
    # W[1, 2]=1 for C[0,1].
    # W[2, 1]=1 for C[1,0].
    # W[3, 0]=1 for C[1,1].
    # So W2 row 0 is C[0,0], row 1 is C[0,1], row 2 is C[1,0], row 3 is C[1,1].
    # This is row-major.
    # We need column-major for the final W.
    # For 4x4, we need to map from row-major (from kron) to column-major (required).
    
    # Permutation for 4x4:
    # Row-major index r = i * 4 + k.
    # Column-major index c = k * 4 + i.
    # We need to reorder rows of W4.
    
    perm = np.zeros(16, dtype=int)
    for i in range(4):
        for k in range(4):
            row_major_idx = i * 4 + k
            col_major_idx = k * 4 + i
            perm[row_major_idx] = col_major_idx
            
    # Apply permutation to W4
    W4_perm = W4[perm, :]
    
    W[0:16, 0:49] = W4_perm
    
    # Part 2: C[0:4, 4:8] = A * B[0:4, 4:8]
    # B part is rows 0-3, cols 4-7 of B.
    # In V, these are rows 0:16 (since B is 4x9, row j of B is j*9 + k? No.)
    # V is (m*p, rank) = (4*9, rank) = (36, rank).
    # V row index for B[j, k] is j * p + k = j * 9 + k.
    # For B[0:4, 4:8], j in 0..3, k in 4..7.
    # Rows in V:
    # j=0, k=4..7 -> 0*9+4=4, 5, 6, 7
    # j=1, k=4..7 -> 1*9+4=13, 14, 15, 16
    # j=2, k=4..7 -> 2*9+4=22, 23, 24, 25
    # j=3, k=4..7 -> 3*9+4=31, 32, 33, 34
    
    # We need to place V4 (16x49) into these rows.
    # V4 corresponds to B 4x4 row-major?
    # In get_strassen_2x2_factors, V is (4, 7).
    # V[0, t] for B[0,0], V[1, t] for B[0,1], V[2, t] for B[1,0], V[3, t] for B[1,1].
    # So V2 is row-major.
    # V4 is row-major for 4x4 B.
    # B 4x4 row-major index: j * 4 + k.
    # We need to map this to the global V index: j * 9 + k.
    
    # Create a mapping from local B 4x4 index (0..15) to global V index (0..35) for cols 4-7.
    v_perm = np.zeros(16, dtype=int)
    for j in range(4):
        for k in range(4):
            local_idx = j * 4 + k
            global_idx = j * 9 + (k + 4) # k+4 because we start at col 4
            v_perm[local_idx] = global_idx
            
    V[v_perm, 49:98] = V4
    
    # W for Part 2: C[0:4, 4:8].
    # i in 0..3, k in 4..7.
    # W row index: k * 4 + i.
    # k=4, i=0 -> 16
    # k=4, i=1 -> 17
    # ...
    # k=7, i=3 -> 31
    
    # We need to place W4_perm (which is for k=0..3) into these rows.
    # The structure of W4_perm is based on local k (0..3).
    # Global k is 4..7.
    # So we just shift the rows.
    # Rows 16..31 in W.
    
    # But wait, W4_perm is already permuted to column-major for k=0..3.
    # The values in W4_perm[row, t] correspond to C[i, k] where row = k*4+i.
    # For the second block, local k' is 0..3, global k is 4+k'.
    # Global row index = (4+k')*4 + i = 16 + k'*4 + i.
    # So we can just copy W4_perm to rows 16..31.
    
    W[16:32, 49:98] = W4_perm
    
    # Part 3: C[0:4, 8:9] = A * B[0:4, 8:9]
    # B part is 4x1.
    # V rows for B[0:4, 8]:
    # j=0, k=8 -> 8
    # j=1, k=8 -> 17
    # j=2, k=8 -> 26
    # j=3, k=8 -> 35
    
    # V1 is (4, 16). Row j corresponds to B[j, 0] (local).
    # So V1[0, t] -> V[8, 98+t]
    # V1[1, t] -> V[17, 98+t]
    # V1[2, t] -> V[26, 98+t]
    # V1[3, t] -> V[35, 98+t]
    
    v1_indices = [8, 17, 26, 35]
    for i in range(4):
        V[v1_indices[i], 98:114] = V1[i, :]
        
    # U for Part 3 is same as U1 (16, 16).
    U[:, 98:114] = U1
    
    # W for Part 3: C[0:4, 8].
    # i in 0..3, k=8.
    # W row index: 8*4 + i = 32 + i.
    # Rows 32, 33, 34, 35.
    
    # W1 is (4, 16). Row i corresponds to C[i, 0] (local).
    # So W1[0, t] -> W[32, 98+t]
    # W1[1, t] -> W[33, 98+t]
    # ...
    
    w1_indices = [32, 33, 34, 35]
    for i in range(4):
        W[w1_indices[i], 98:114] = W1[i, :]
        
    # Verify
    if verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

def verify(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # Compute S_ijk = sum_t U_ijt V_jkt W_kit
    # Using einsum
    S = np.einsum('ijt,jkt,kit->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    if not np.allclose(S, 1.0, atol=1e-9):
        return False
    return True
