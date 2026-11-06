import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # We construct a rank-189 decomposition using Strassen (rank 7) for the 2x2 block structure
    # and Schoolbook (rank 27) for the 3x3 internal structure.
    # This is the Kronecker product of the M_2,2,2 (Strassen) and M_3,3,3 (Schoolbook) tensors.
    # Rank = 7 * 27 = 189.
    
    n, m, p = DIMENSIONS
    
    # 1. Define Strassen decomposition for 2x2 (Rank 7)
    # U_22, V_22, W_22 of shape (4, 7)
    # Indices: i*2+j, j*2+k, k*2+i
    U_22 = np.zeros((4, 7), dtype=int)
    V_22 = np.zeros((4, 7), dtype=int)
    W_22 = np.zeros((4, 7), dtype=int)
    
    # Strassen formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # U (A factors):
    # M1: A11, A22
    # M2: A21, A22
    # M3: A11
    # M4: A22
    # M5: A11, A12
    # M6: A21, A11
    # M7: A12, A22
    
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    U_22[0, 0] = 1; U_22[3, 0] = 1 # M1: A00, A11
    U_22[2, 1] = 1; U_22[3, 1] = 1 # M2: A10, A11
    U_22[0, 2] = 1                 # M3: A00
    U_22[3, 3] = 1                 # M4: A11
    U_22[0, 4] = 1; U_22[1, 4] = 1 # M5: A00, A01
    U_22[2, 5] = 1; U_22[0, 5] = -1# M6: A10, -A00
    U_22[1, 6] = 1; U_22[3, 6] = -1# M7: A01, -A11
    
    # V (B factors):
    # M1: B11, B22
    # M2: B11
    # M3: B21, B22
    # M4: B12, B11
    # M5: B22
    # M6: B11, B12
    # M7: B21, B22
    
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    V_22[0, 0] = 1; V_22[3, 0] = 1 # M1: B00, B11
    V_22[0, 1] = 1                 # M2: B00
    V_22[2, 2] = 1; V_22[3, 2] = -1# M3: B10, -B11
    V_22[1, 3] = 1; V_22[0, 3] = -1# M4: B01, -B00
    V_22[3, 4] = 1                 # M5: B11
    V_22[0, 5] = 1; V_22[1, 5] = 1 # M6: B00, B01
    V_22[2, 6] = 1; V_22[3, 6] = 1 # M7: B10, B11
    
    # W (C factors):
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    W_22[0, 0] = 1; W_22[0, 3] = 1; W_22[0, 4] = -1; W_22[0, 6] = 1
    W_22[1, 2] = 1; W_22[1, 4] = 1
    W_22[2, 1] = 1; W_22[2, 3] = 1
    W_22[3, 0] = 1; W_22[3, 1] = -1; W_22[3, 2] = 1; W_22[3, 5] = 1

    # 2. Define Schoolbook decomposition for 3x3 (Rank 27)
    # U_33, V_33, W_33 of shape (9, 27)
    U_33 = np.zeros((9, 27), dtype=int)
    V_33 = np.zeros((9, 27), dtype=int)
    W_33 = np.zeros((9, 27), dtype=int)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_33[i * 3 + j, col] = 1
                V_33[j * 3 + k, col] = 1
                W_33[k * 3 + i, col] = 1
                col += 1

    # 3. Construct 6x6 decomposition via Kronecker Product
    # Rank = 7 * 27 = 189
    rank = 189
    U_66 = np.zeros((36, rank), dtype=int)
    V_66 = np.zeros((36, rank), dtype=int)
    W_66 = np.zeros((36, rank), dtype=int)
    
    # Mapping of flat indices to (outer, inner) indices
    # Outer: 2x2, Inner: 3x3
    # Flat Index R = i_out * 3 + i_in ? No.
    # The tensor is T_22 \otimes T_33.
    # Indices I = i_2 * 3 + i_3.
    # U_66[I * 6 + J, C] = U_22[i_2 * 2 + j_2, c_2] * U_33[i_3 * 3 + j_3, c_3]
    
    # We iterate over all flat row indices R (0..35) and rank indices C (0..188)
    for R in range(36):
        # Decompose R into I, J
        I = R // 6
        J = R % 6
        
        # Decompose I into i_2, i_3
        i_2 = I // 3
        i_3 = I % 3
        
        # Decompose J into j_2, j_3
        j_2 = J // 3
        j_3 = J % 3
        
        r_2 = i_2 * 2 + j_2
        r_3 = i_3 * 3 + j_3
        
        for C in range(rank):
            c_2 = C // 27
            c_3 = C % 27
            
            U_66[R, C] = U_22[r_2, c_2] * U_33[r_3, c_3]
            
            # V_66
            # V index is J * 6 + K
            # We need to fill V_66 for all rows. 
            # It is more efficient to fill V_66 in a separate loop or within the same if we can map.
            # Since V depends on J and K, and we are iterating R (I, J), we can fill V_66[R, C] 
            # but V_66 row index is J*6+K.
            # So for a given R (which fixes I, J), we need to iterate K to fill V_66.
            # This is inefficient.
            
    # Re-do the construction more efficiently
    U_66 = np.zeros((36, rank), dtype=int)
    V_66 = np.zeros((36, rank), dtype=int)
    W_66 = np.zeros((36, rank), dtype=int)
    
    # Precompute mappings
    # For U: Row R = I*6+J. I=i2*3+i3, J=j2*3+j3.
    # r2 = i2*2+j2, r3 = i3*3+j3.
    # For V: Row R = J*6+K. J=j2*3+j3, K=k2*3+k3.
    # r2 = j2*2+k2, r3 = j3*3+k3.
    # For W: Row R = K*6+I. K=k2*3+k3, I=i2*3+i3.
    # r2 = k2*2+i2, r3 = k3*3+i3.
    
    for R in range(36):
        # U
        I = R // 6
        J = R % 6
        i2, i3 = divmod(I, 3)
        j2, j3 = divmod(J, 3)
        r2_u = i2 * 2 + j2
        r3_u = i3 * 3 + j3
        
        # V
        # Row R for V is J*6+K.
        # We are iterating R as the row index for U.
        # We need to fill V_66 at row index J*6+K for all K.
        # This loop structure is bad for V and W.
        
    # Let's just iterate over all indices for U, V, W separately.
    
    for R in range(36):
        I = R // 6
        J = R % 6
        i2, i3 = divmod(I, 3)
        j2, j3 = divmod(J, 3)
        r2 = i2 * 2 + j2
        r3 = i3 * 3 + j3
        for C in range(rank):
            c2 = C // 27
            c3 = C % 27
            U_66[R, C] = U_22[r2, c2] * U_33[r3, c3]
            
    for R in range(36):
        J = R // 6
        K = R % 6
        j2, j3 = divmod(J, 3)
        k2, k3 = divmod(K, 3)
        r2 = j2 * 2 + k2
        r3 = j3 * 3 + k3
        for C in range(rank):
            c2 = C // 27
            c3 = C % 27
            V_66[R, C] = V_22[r2, c2] * V_33[r3, c3]
            
    for R in range(36):
        K = R // 6
        I = R % 6
        k2, k3 = divmod(K, 3)
        i2, i3 = divmod(I, 3)
        r2 = k2 * 2 + i2
        r3 = k3 * 3 + i3
        for C in range(rank):
            c2 = C // 27
            c3 = C % 27
            W_66[R, C] = W_22[r2, c2] * W_33[r3, c3]

    return {"U": U_66.tolist(), "V": V_66.tolist(), "W": W_66.tolist()}
# EVOLVE-BLOCK-END
