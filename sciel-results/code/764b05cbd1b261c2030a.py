import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Strassen decomposition for 2x2 (Rank 7)
    # U2: (4, 7), V2: (4, 7), W2: (4, 7)
    # Indices: U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[1, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    
    # M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = -1
    
    # M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1; W2[3, 3] = -1
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = 1; W2[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[2, 5] = 1; W2[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = -1; W2[3, 6] = -1

    # 2. Laderman decomposition for 3x3 (Rank 7)
    # U3: (9, 7), V3: (9, 7), W3: (9, 7)
    # Indices: U3[i*3+j, r], V3[j*3+k, r], W3[k*3+i, r]
    
    U3 = np.zeros((9, 7), dtype=float)
    V3 = np.zeros((9, 7), dtype=float)
    W3 = np.zeros((9, 7), dtype=float)
    
    # Laderman's 7 products:
    # M1 = A00 * B00
    U3[0, 0] = 1
    V3[0, 0] = 1
    W3[0, 0] = 1
    
    # M2 = (A00 + A11) * (B00 + B11)
    U3[0, 1] = 1; U3[4, 1] = 1
    V3[0, 1] = 1; V3[4, 1] = 1
    W3[0, 1] = 1; W3[4, 1] = 1
    
    # M3 = A11 * B11
    U3[4, 2] = 1
    V3[4, 2] = 1
    W3[4, 2] = 1
    
    # M4 = (A01 + A12) * (B10 + B21)
    U3[1, 3] = 1; U3[2, 3] = 1
    V3[1, 3] = 1; V3[5, 3] = 1
    W3[1, 3] = 1; W3[2, 3] = 1
    
    # M5 = (A20 + A02) * (B02 + B20)
    U3[2, 4] = 1; U3[6, 4] = 1
    V3[2, 4] = 1; V3[6, 4] = 1
    W3[2, 4] = 1; W3[6, 4] = 1
    
    # M6 = (A10 - A02 + A22) * (B20 + B02 - B22)
    U3[3, 5] = 1; U3[6, 5] = -1; U3[8, 5] = 1
    V3[6, 5] = 1; V3[2, 5] = 1; V3[8, 5] = -1
    W3[0, 5] = 1; W3[8, 5] = 1; W3[6, 5] = -1
    
    # M7 = (A01 - A20 + A21) * (B10 + B20 - B12)
    U3[1, 6] = 1; U3[6, 6] = -1; U3[7, 6] = 1
    V3[1, 6] = 1; V3[3, 6] = 1; V3[5, 6] = -1
    W3[0, 6] = 1; W3[6, 6] = 1; W3[1, 6] = -1

    # 3. Kronecker Product to form 6x6 decomposition (Rank 49)
    # The tensor product of the 2x2 and 3x3 multiplications gives a 6x6 multiplication.
    # The indices for the 6x6 case are (i*6 + j) where i in {0,1} (from 2x2) and j in {0,1,2} (from 3x3).
    # Specifically, if A is 6x6, A[i,j] corresponds to A2[i2, i3] * A3[j2, j3] where i = i2*3 + i3?
    # Wait, the standard Kronecker product ordering is:
    # (A \otimes B)[i*3 + j, k*3 + l] = A[i,k] * B[j,l]
    # Here A is 2x2, B is 3x3.
    # So the 6x6 matrix C = A2 \otimes A3.
    # C[i*3 + j, k*3 + l] = A2[i,k] * A3[j,l].
    
    # The rank is 7 * 7 = 49.
    # The rank indices are (r2, r3) where r2 in {0..6}, r3 in {0..6}.
    # Let r = r2 * 7 + r3.
    
    # U6 shape: (36, 49)
    # U6[i*6 + j, r] = U2[i2, r2] * U3[i3, r3]
    # where i = i2 * 3 + i3, j = j2 * 3 + j3?
    # No, the row index of U6 is the index of the element in the 6x6 matrix A.
    # The element A[i, j] in the 6x6 matrix corresponds to A2[i2, i3] * A3[j2, j3]?
    # Let's check the index mapping.
    # A6[i, j] where i, j in 0..5.
    # i = i2 * 3 + i3, j = j2 * 3 + j3.
    # Then A6[i, j] = A2[i2, j2] * A3[i3, j3].
    # So U6[i, r] = U2[i2, r2] * U3[i3, r3].
    
    U6 = np.zeros((36, 49), dtype=float)
    V6 = np.zeros((36, 49), dtype=float)
    W6 = np.zeros((36, 49), dtype=float)
    
    for i2 in range(2):
        for i3 in range(3):
            i = i2 * 3 + i3
            for j2 in range(2):
                for j3 in range(3):
                    j = j2 * 3 + j3
                    for r2 in range(7):
                        for r3 in range(7):
                            r = r2 * 7 + r3
                            val_u = U2[i2, r2] * U3[i3, r3]
                            val_v = V2[j2, r2] * V3[j3, r3]
                            val_w = W2[i2, r2] * W3[i3, r3] # Wait, W index is (k, i)
                            # W6[k, r] = W2[k2, r2] * W3[k3, r3]
                            # k = k2 * 3 + k3
                            # So we need to iterate over k2, k3 for W.
                            # But U and V depend on i, j.
                            # Let's restructure the loop to be clearer.
                            
    # Let's recompute U6, V6, W6 properly.
    U6 = np.zeros((36, 49), dtype=float)
    V6 = np.zeros((36, 49), dtype=float)
    W6 = np.zeros((36, 49), dtype=float)
    
    for r2 in range(7):
        for r3 in range(7):
            r = r2 * 7 + r3
            for i2 in range(2):
                for i3 in range(3):
                    i = i2 * 3 + i3
                    U6[i, r] = U2[i2, r2] * U3[i3, r3]
            for j2 in range(2):
                for j3 in range(3):
                    j = j2 * 3 + j3
                    V6[j, r] = V2[j2, r2] * V3[j3, r3]
            for k2 in range(2):
                for k3 in range(3):
                    k = k2 * 3 + k3
                    W6[k, r] = W2[k2, r2] * W3[k3, r3]

    # 4. Extract 5x5 from 6x6
    # We need to select the first 5 rows and columns of the 6x6 matrix.
    # The indices for 5x5 are 0, 1, 2, 3, 4.
    # We select rows 0..4 from U6, V6, W6.
    # Note: The "output factor" W[k*n+i, r] in the 5x5 case corresponds to W6[k*6+i, r]
    # BUT ONLY IF the 5x5 matrix is a principal submatrix of the 6x6 matrix.
    # The 5x5 matrix A5 is the top-left 5x5 block of A6.
    # A5[i, j] = A6[i, j] for i, j in 0..4.
    # So U5[i, r] = U6[i, r] for i in 0..4.
    # V5[j, r] = V6[j, r] for j in 0..4.
    # W5[k, i, r] -> W5[k*5+i, r].
    # In the 6x6 case, the element C6[k, i] = sum_r U6[i, r] V6[k, r] W6[k, i, r]?
    # No, W6 is the output factor. W6[k*6+i, r].
    # We need W5[k*5+i, r] such that C5[k, i] = sum_r U5[i, r] V5[k, r] W5[k*5+i, r].
    # Since C5 is the top-left 5x5 block of C6, C5[k, i] = C6[k, i].
    # C6[k, i] = sum_r U6[i, r] V6[k, r] W6[k*6+i, r].
    # So we can simply take W5[k*5+i, r] = W6[k*6+i, r].
    
    U5 = U6[:25, :] # Rows 0..24 correspond to i=0..4, j=0..4?
    # Wait. U6 has 36 rows, indexed by i (row of A).
    # A is 6x6. The rows of A are 0..5.
    # We only need rows 0..4 of A.
    # So U5 = U6[:5, :]? No, U is (n*m, rank).
    # For 5x5, n=5, m=5. U is (25, rank).
    # The index of U is i*m + j.
    # In the 6x6 case, U6 index is i*6 + j.
    # We need to map the 5x5 indices (i, j) with i, j in 0..4 to the 6x6 indices.
    # A5[i, j] = A6[i, j].
    # So U5[i*5 + j, r] = U6[i*6 + j, r].
    
    U5 = np.zeros((25, 49), dtype=float)
    V5 = np.zeros((25, 49), dtype=float)
    W5 = np.zeros((25, 49), dtype=float)
    
    for i in range(5):
        for j in range(5):
            U5[i*5 + j, :] = U6[i*6 + j, :]
            V5[i*5 + j, :] = V6[i*6 + j, :]
    
    for k in range(5):
        for i in range(5):
            W5[k*5 + i, :] = W6[k*6 + i, :]
    
    # Convert to lists of lists
    U_list = U5.tolist()
    V_list = V5.tolist()
    W_list = W5.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
