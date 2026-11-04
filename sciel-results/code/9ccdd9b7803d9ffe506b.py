import numpy as np

def construct():
    # Strassen's 7-term decomposition for 2x2 matrices
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices for U2, V2 are row*2+col of the input matrices A, B
    # Index for W2 is row*2+col of the output matrix C, where C = A*B
    # Note: C[i,j] = sum_k A[i,k] B[k,j].
    # The task requires W[k*n + i, r].
    # For 2x2, n=2. W index is k*2 + i.
    # Let's verify the indices for W2 based on previous derivation.
    # C[0,0] index 0. C[0,1] index 1. C[1,0] index 2. C[1,1] index 3.
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # r=0: M1 = (A00 + A11)(B00 + B11)
    # A00 idx 0, A11 idx 3
    U2[0, 0] = 1
    U2[3, 0] = 1
    # B00 idx 0, B11 idx 3
    V2[0, 0] = 1
    V2[3, 0] = 1
    # C00 idx 0 (+1), C11 idx 3 (+1)
    W2[0, 0] = 1
    W2[3, 0] = 1
    
    # r=1: M2 = (A10 + A11)B00
    # A10 idx 2, A11 idx 3
    U2[2, 1] = 1
    U2[3, 1] = 1
    # B00 idx 0
    V2[0, 1] = 1
    # C10 idx 2 (+1), C11 idx 3 (-1)
    W2[2, 1] = 1
    W2[3, 1] = -1
    
    # r=2: M3 = A00(B10 - B11)
    # A00 idx 0
    U2[0, 2] = 1
    # B10 idx 2, B11 idx 3
    V2[2, 2] = 1
    V2[3, 2] = -1
    # C01 idx 1 (+1), C11 idx 3 (+1)
    W2[1, 2] = 1
    W2[3, 2] = 1
    
    # r=3: M4 = A11(B01 - B00)
    # A11 idx 3
    U2[3, 3] = 1
    # B01 idx 1, B00 idx 0
    V2[1, 3] = 1
    V2[0, 3] = -1
    # C00 idx 0 (+1), C10 idx 2 (+1)
    W2[0, 3] = 1
    W2[2, 3] = 1
    
    # r=4: M5 = (A00 + A01)B11
    # A00 idx 0, A01 idx 1
    U2[0, 4] = 1
    U2[1, 4] = 1
    # B11 idx 3
    V2[3, 4] = 1
    # C00 idx 0 (-1), C01 idx 1 (+1)
    W2[0, 4] = -1
    W2[1, 4] = 1
    
    # r=5: M6 = (A10 - A00)(B00 + B10)
    # A10 idx 2, A00 idx 0
    U2[2, 5] = 1
    U2[0, 5] = -1
    # B00 idx 0, B10 idx 2
    V2[0, 5] = 1
    V2[2, 5] = 1
    # C11 idx 3 (+1)
    W2[3, 5] = 1
    
    # r=6: M7 = (A01 - A11)(B01 + B11)
    # A01 idx 1, A11 idx 3
    U2[1, 6] = 1
    U2[3, 6] = -1
    # B01 idx 1, B11 idx 3
    V2[1, 6] = 1
    V2[3, 6] = 1
    # C00 idx 0 (+1)
    W2[0, 6] = 1

    # Kronecker product to get 4x4 factors
    # U4 shape (16, 49)
    # V4 shape (16, 49)
    # W4 shape (16, 49)
    
    # We use numpy's kron for simplicity and correctness, then reshape/permute if needed
    # But standard kron(A, B) has row index i_A * rows_B + i_B.
    # Our U4 row index is i*4 + j.
    # i = i1*2 + i0, j = j1*2 + j0.
    # Index = (i1*2 + i0)*4 + (j1*2 + j0) = i1*8 + i0*4 + j1*2 + j0.
    # Kron(U2, U2) row index: (i1*2 + j1)*4 + (i0*2 + j0) = i1*8 + j1*4 + i0*2 + j0.
    # The ordering is different. We need to permute the rows.
    
    # Let's construct directly to avoid confusion.
    n = 4
    m = 4
    p = 4
    rank = 49
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for i1 in range(2):
        for i0 in range(2):
            for j1 in range(2):
                for j0 in range(2):
                    row_u = i1 * 4 + i0 * 2 + j1 * 1 + j0 # Wait, i*4+j. i=i1*2+i0. j=j1*2+j0.
                    # row_u = (i1*2 + i0)*4 + (j1*2 + j0) = i1*8 + i0*4 + j1*2 + j0
                    row_u = i1 * 8 + i0 * 4 + j1 * 2 + j0
                    
                    for r1 in range(7):
                        for r0 in range(7):
                            col_r = r1 * 7 + r0
                            
                            # U contribution
                            val_u = U2[i1 * 2 + j1, r1] * U2[i0 * 2 + j0, r0]
                            U4[row_u, col_r] = val_u
                            
                            # V contribution
                            # V row index: j*4 + k.
                            # We need to sum over k? No, V4 is a matrix.
                            # V4[j, k, r] corresponds to V[j*4+k, r].
                            # j = j1*2+j0, k = k1*2+k0.
                            # row_v = (j1*2+j0)*4 + (k1*2+k0) = j1*8 + j0*4 + k1*2 + k0
                            # We are iterating over j1, j0. We need to also iterate over k1, k0 for V4?
                            # Yes, V4 has 16 rows.
                            
                            # W contribution
                            # W row index: k*4 + i.
                            # k = k1*2+k0, i = i1*2+i0.
                            # row_w = (k1*2+k0)*4 + (i1*2+i0) = k1*8 + k0*4 + i1*2 + i0
                            # We need to iterate over k1, k0 for W4.
                            
                    # It is more efficient to fill V4 and W4 in their own loops or nested loops.
                    # Let's restart the filling logic to be cleaner.
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    # Fill U4
    for i1 in range(2):
        for i0 in range(2):
            for j1 in range(2):
                for j0 in range(2):
                    row_u = i1 * 8 + i0 * 4 + j1 * 2 + j0
                    for r1 in range(7):
                        for r0 in range(7):
                            col_r = r1 * 7 + r0
                            U4[row_u, col_r] = U2[i1 * 2 + j1, r1] * U2[i0 * 2 + j0, r0]
                            
    # Fill V4
    for j1 in range(2):
        for j0 in range(2):
            for k1 in range(2):
                for k0 in range(2):
                    row_v = j1 * 8 + j0 * 4 + k1 * 2 + k0
                    for r1 in range(7):
                        for r0 in range(7):
                            col_r = r1 * 7 + r0
                            V4[row_v, col_r] = V2[j1 * 2 + k1, r1] * V2[j0 * 2 + k0, r0]
                            
    # Fill W4
    for k1 in range(2):
        for k0 in range(2):
            for i1 in range(2):
                for i0 in range(2):
                    row_w = k1 * 8 + k0 * 4 + i1 * 2 + i0
                    for r1 in range(7):
                        for r0 in range(7):
                            col_r = r1 * 7 + r0
                            W4[row_w, col_r] = W2[k1 * 2 + i1, r1] * W2[k0 * 2 + i0, r0]

    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
