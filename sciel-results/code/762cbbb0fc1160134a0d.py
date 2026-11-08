import numpy as np

def construct():
    # Dimensions
    n, m, p = 4, 4, 5
    
    # Part 1: 4x4x4 matrix multiplication using Strassen (Rank 49)
    # First, define the 2x2x2 Strassen factors
    # U, V, W shapes: 4x7
    # Indices: U[a,b], V[b,c], W[c,a]
    
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    U2[0, 0] = 1; U2[3, 0] = 1  # A00, A11
    V2[0, 0] = 1; V2[3, 0] = 1  # B00, B11
    W2[0, 0] = 1; W2[3, 0] = 1  # C00, C11
    
    # M2: (A10+A11)B00 -> C10 += 1, C11 += -1
    U2[2, 1] = 1; U2[3, 1] = 1  # A10, A11
    V2[0, 1] = 1                # B00
    W2[1, 1] = 1; W2[3, 1] = -1 # C10, C11
    
    # M3: A00(B10-B11) -> C01 += 1, C11 += 1
    U2[0, 2] = 1                # A00
    V2[2, 2] = 1; V2[3, 2] = -1 # B10, B11
    W2[1, 2] = 1; W2[3, 2] = 1  # C01, C11
    
    # M4: A11(B01-B00) -> C10 += 1
    U2[3, 3] = 1                # A11
    V2[1, 3] = 1; V2[0, 3] = -1 # B01, B00
    W2[1, 3] = 1                # C10
    
    # M5: (A00+A01)B11 -> C00 += -1, C01 += 1
    U2[0, 4] = 1; U2[1, 4] = 1  # A00, A01
    V2[3, 4] = 1                # B11
    W2[0, 4] = -1; W2[1, 4] = 1 # C00, C01
    
    # M6: (A10-A00)(B00+B10) -> C11 += 1
    U2[2, 5] = 1; U2[0, 5] = -1 # A10, A00
    V2[0, 5] = 1; V2[2, 5] = 1  # B00, B10
    W2[3, 5] = 1                # C11
    
    # M7: (A01-A11)(B10+B11) -> C00 += 1
    U2[1, 6] = 1; U2[3, 6] = -1 # A01, A11
    V2[2, 6] = 1; V2[3, 6] = 1  # B10, B11
    W2[0, 6] = 1                # C00
    
    # Now construct 4x4x4 factors
    # Rank 49 = 7 * 7
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for alpha1 in range(7):
        for alpha0 in range(7):
            col = alpha1 * 7 + alpha0
            for i0 in range(2):
                for i1 in range(2):
                    for j0 in range(2):
                        for j1 in range(2):
                            i = i1 * 2 + i0
                            j = j1 * 2 + j0
                            u_val = U2[i0 * 2 + j0, alpha0] * U2[i1 * 2 + j1, alpha1]
                            U4[i * 4 + j, col] = u_val
                            
                            v_val = V2[j0 * 2 + (j0*2+j0)%2, alpha0] # Wait, V indices are (b,c). b=j, c=k.
                            # In 2x2, V[b,c]. b=j0, c=k0.
                            # In 4x4, V[j,k]. j=j1*2+j0, k=k1*2+k0.
                            # We need to sum over k too.
                            # The term is product of two 2x2 multiplications.
                            # M = (A1 A0) (B1 B0)
                            # The 7 terms for block 1 (indices 1) and block 0 (indices 0).
                            # U4[i,j] = sum U2[i0,j0,a0] U2[i1,j1,a1]
                            # V4[j,k] = sum V2[j0,k0,a0] V2[j1,k1,a1]
                            # W4[k,i] = sum W2[k0,i0,a0] W2[k1,i1,a1]
                            
                            # So we must iterate over k as well for V and W construction, 
                            # but U only depends on i,j.
                            # Let's rebuild the loop to handle all indices.
            pass # The previous loop was incomplete for V and W. Let's restart the construction part properly.
            
    # Correct construction of 4x4x4 factors
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for alpha1 in range(7):
        for alpha0 in range(7):
            col = alpha1 * 7 + alpha0
            for i0 in range(2):
                for i1 in range(2):
                    for j0 in range(2):
                        for j1 in range(2):
                            i = i1 * 2 + i0
                            j = j1 * 2 + j0
                            U4[i * 4 + j, col] = U2[i0 * 2 + j0, alpha0] * U2[i1 * 2 + j1, alpha1]
                            
                            for k0 in range(2):
                                for k1 in range(2):
                                    k = k1 * 2 + k0
                                    V4[j * 4 + k, col] += V2[j0 * 2 + k0, alpha0] * V2[j1 * 2 + k1, alpha1]
                                    W4[k * 4 + i, col] += W2[k0 * 2 + i0, alpha0] * W2[k1 * 2 + i1, alpha1]
                                    
    # Part 2: 4x4x1 matrix multiplication (Schoolbook, Rank 16)
    # C[:, 4] = A @ B[:, 4]
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    U_ext = np.zeros((16, 16), dtype=float)
    V_ext = np.zeros((20, 16), dtype=float) # m*p = 4*5 = 20
    W_ext = np.zeros((20, 16), dtype=float) # p*n = 5*4 = 20
    
    col_ext = 0
    for i in range(4):
        for j in range(4):
            U_ext[i * 4 + j, col_ext] = 1
            V_ext[j * 5 + 4, col_ext] = 1
            W_ext[4 * 4 + i, col_ext] = 1
            col_ext += 1
            
    # Combine
    U_final = np.hstack([U4, U_ext])
    V_final = np.hstack([V4, V_ext])
    W_final = np.hstack([W4, W_ext])
    
    # Convert to lists of lists as required
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
