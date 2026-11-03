import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    t0 = time.time()
    deadline = t0 + 110  # safety margin before 120s limit
    
    # Strassen 2x2x2 factors
    # U: (4,7) indexed by i*2+j, V: (4,7) indexed by j*2+k, W: (4,7) indexed by k*2+i
    def strassen_2x2():
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1
        
        # M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3 = A11(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4 = A22(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1
        
        # M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1
        
        return U, V, W
    
    # Verify Strassen 2x2x2
    U2, V2, W2 = strassen_2x2()
    # Check: for each (i,j,k), sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r] should be 1 if i,j,k form valid product
    # Actually the tensor T[i,j,k] = delta_{i,k} (for matmul C[i,k] = sum_j A[i,j]B[j,k])
    # Wait, for 2x2x2: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor index: T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i,j,k] = 1 for all valid (i,j,k) since each is one term
    # Actually no - the tensor is T[i,j,k] where the multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has indices (i,j,k) and T[i,j,k] = 1 for all i,j,k in {0,1}
    # Wait that's not right either. Let me think again.
    
    # For matrix multiplication C = A*B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) with T[i,j,k] = 1 for all valid i,j,k
    # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Wait, W index is (k,i) not (i,k). Let me re-read the task.
    
    # "W[k*n+i,r]" - so W is indexed by (k,i) where k is column of C, i is row of C
    # So W index = k*n + i
    
    # For 2x2x2: n=2, m=2, p=2
    # U index: i*2+j (0..3)
    # V index: j*2+k (0..3)  
    # W index: k*2+i (0..3)
    
    # T[i,j,k] = 1 for all i,j,k in {0,1} (each term A[i,j]*B[j,k] appears exactly once)
    # Check: sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r] should equal 1 for all i,j,k
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = np.sum(U2[i*2+j, :] * V2[j*2+k, :] * W2[k*2+i, :])
                if abs(val - 1.0) > 1e-10:
                    print(f"Strassen 2x2 error at ({i},{j},{k}): {val}")
                    # Fall back to schoolbook
                    rank = n * m * p
                    U = np.zeros((n*m, rank), dtype=float)
                    V = np.zeros((m*p, rank), dtype=float)
                    W = np.zeros((p*n, rank), dtype=float)
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii*m+jj, col] = 1
                                V[jj*p+kk, col] = 1
                                W[kk*n+ii, col] = 1
                                col += 1
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now construct 4x4x4 using Strassen on 2x2 blocks
    # A is 4x4, split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # B is 4x4, split into 2x2 blocks:
    # B = [[B00, B01], [B10, B11]]
    # C is 4x4, split into 2x2 blocks:
    # C = [[C00, C01], [C10, C11]]
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M6
    
    # Each Mk is a 2x2 matrix product, decomposable with rank 7
    # Total rank for 4x4x4 = 7 * 7 = 49
    
    # For each of the 7 Strassen multiplications Mk = Xk * Yk (2x2 * 2x2):
    # Xk is a linear combination of A blocks
    # Yk is a linear combination of B blocks
    # The 2x2x2 Strassen decomposition gives us 7 terms per Mk
    # Each term: U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]
    
    # For the 4x4x4 tensor:
    # U4 index: I*4+J where I,J in 0..3
    # V4 index: J*4+K where J,K in 0..3
    # W4 index: K*4+I where K,I in 0..3
    
    # For Mk = Xk * Yk where Xk = sum_{(r,c)} coeff_X[r,c] * A_{r,c}
    # and Yk = sum_{(r,c)} coeff_Y[r,c] * B_{r,c}
    
    # The contribution to U4 from term (i,j,r) in the 2x2 Strassen for Mk:
    # U4[I*4+J, col] += coeff_X[block_row, block_col] * U2[i*2+j, r]
    # where I = 2*block_row + i, J = 2*block_col + j
    
    # Similarly for V4 and W4.
    
    # Define the Strassen coefficients for each Mk:
    # Mk: (coeffs for A blocks, coeffs for B blocks, coeffs for C blocks)
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # B blocks: (0,0), (0,1), (1,0), (1,1)
    # C blocks: (0,0), (0,1), (1,0), (1,1)
    
    strassen_defs = [
        # Mk: (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: dict from (row_block, col_block) to coefficient
        # M1 = (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, 
         {(0,0): 1, (1,1): 1, (0,1): 0, (1,0): 0}),  # C00 += M1, C11 += M1
        # M2 = (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, 
         {(0,0): 0, (1,1): 0, (0,1): 0, (1,0): 1}),  # C10 += M2
        # M3 = A00(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, 
         {(0,0): 0, (1,1): 0, (0,1): 1, (1,0): 0}),  # C01 += M3
        # M4 = A11(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, 
         {(0,0): 0, (1,1): 0, (0,1): 0, (1,0): 1}),  # C10 += M4
        # M5 = (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, 
         {(0,0): 0, (1,1): 0, (0,1): 1, (1,0): 0}),  # C01 += M5
        # M6 = (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, 
         {(0,0): 0, (1,1): 1, (0,1): 0, (1,0): 0}),  # C11 += M6
        # M7 = (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, 
         {(0,0): 0, (1,1): 1, (0,1): 0, (1,0): 0}),  # C11 += M7
    ]
    
    # Wait, I need to be more careful. The C coefficients tell us how Mk contributes to each C block.
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M6
    
    # So for each Mk, it contributes to specific C blocks with specific signs.
    # Let me redo this:
    
    strassen_defs = [
        # Mk: (A_coeffs, B_coeffs, C_coeffs)
        # M1 = (A00+A11)(B00+B11), contributes to C00 with +1, C11 with +1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, 
         {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
        # M2 = (A10+A11)B00, contributes to C10 with +1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, 
         {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0}),
        # M3 = A00(B01-B11), contributes to C01 with +1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, 
         {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0}),
        # M4 = A11(B10-B00), contributes to C10 with +1
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, 
         {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0}),
        # M5 = (A00+A01)B11, contributes to C01 with +1
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, 
         {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0}),
        # M6 = (A10-A00)(B00+B01), contributes to C11 with +1
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, 
         {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
        # M7 = (A01-A11)(B10+B11), contributes to C11 with +1
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, 
         {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
    ]
    
    # Hmm wait, that's not right. Let me re-derive.
    # C00 = M1 - M4 - M5 + M7
    # So M1 contributes +1 to C00, M4 contributes -1 to C00, M5 contributes -1 to C00, M7 contributes +1 to C00
    # C01 = M3 + M5
    # M3 contributes +1 to C01, M5 contributes +1 to C01
    # C10 = M2 + M4
    # M2 contributes +1 to C10, M4 contributes +1 to C10
    # C11 = M1 - M2 - M3 + M6
    # M1 contributes +1 to C11, M2 contributes -1 to C11, M3 contributes -1 to C11, M6 contributes +1 to C11
    
    strassen_defs = [
        # M1 = (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, 
         {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
        # M2 = (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, 
         {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
        # M3 = A00(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, 
         {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): -1}),
        # M4 = A11(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, 
         {(0,0): -1, (0,1): 0, (1,0): 1, (1,1): 0}),
        # M5 = (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, 
         {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
        # M6 = (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, 
         {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
        # M7 = (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, 
         {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
    ]
    
    # Now build U4, V4, W4 for 4x4x4 (rank 49)
    rank4 = 49
    U4 = np.zeros((16, rank4), dtype=float)
    V4 = np.zeros((16, rank4), dtype=float)
    W4 = np.zeros((16, rank4), dtype=float)
    
    for mk in range(7):
        A_coeffs, B_coeffs, C_coeffs = strassen_defs[mk]
        for r in range(7):  # 7 terms in Strassen 2x2x2
            col = mk * 7 + r
            for i2 in range(2):
                for j2 in range(2):
                    for k2 in range(2):
                        # U2[i2*2+j2, r] * V2[j2*2+k2, r] * W2[k2*2+i2, r]
                        u_val = U2[i2*2+j2, r]
                        v_val = V2[j2*2+k2, r]
                        w_val = W2[k2*2+i2, r]
                        
                        if abs(u_val) < 1e-15 and abs(v_val) < 1e-15 and abs(w_val) < 1e-15:
                            continue
                        
                        # For U4: index I*4+J where I = 2*br + i2, J = 2*bc + j2
                        # br, bc are the block row/col in A
                        for (br, bc), ca in A_coeffs.items():
                            I = 2*br + i2
                            J = 2*bc + j2
                            U4[I*4+J, col] += ca * u_val
                        
                        # For V4: index J*4+K where J = 2*br + j2, K = 2*bc + k2
                        # br, bc are the block row/col in B
                        for (br, bc), cb in B_coeffs.items():
                            J = 2*br + j2
                            K = 2*bc + k2
                            V4[J*4+K, col] += cb * v_val
                        
                        # For W4: index K*4+I where K = 2*br + k2, I = 2*bc + i2
                        # br, bc are the block row/col in C
                        for (br, bc), cc in C_coeffs.items():
                            K = 2*br + k2
                            I = 2*bc + i2
                            W4[K*4+I, col] += cc * w_val
    
    # Verify 4x4x4 decomposition
    valid4 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = np.sum(U4[i*4+j, :] * V4[j*4+k, :] * W4[k*4+i, :])
                if abs(val - 1.0) > 1e-8:
                    valid4 = False
                    break
            if not valid4:
                break
        if not valid4:
            break
    
    if not valid4:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U[ii*m+jj, col] = 1
                    V[jj*p+kk, col] = 1
                    W[kk*n+ii, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now extend to 4x4x8
    # B is 4x8, split into B_left (4x4) and B_right (4x4)
    # C = A * B = [A*B_left, A*B_right]
    # C is 4x8, so C[i, k] for k in 0..7
    # C_left = A * B_left (4x4), C_right = A * B_right (4x4)
    
    # For 4x4x8:
    # U: (16, rank), V: (32, rank), W: (32, rank)
    # U index: i*4+j, V index: j*8+k, W index: k*4+i
    
    # The left half of C (k=0..3) uses B_left (j*4+k for k=0..3)
    # The right half of C (k=4..7) uses B_right (j*4+k for k=4..7)
    
    # For the left half: same as 4x4x4 but V index is j*4+k (k=0..3)
    # For the right half: same as 4x4x4 but V index is j*4+(k+4) = j*8+(k+4)
    
    # Total rank = 49 + 49 = 98
    rank8 = 98
    U8 = np.zeros((16, rank8), dtype=float)
    V8 = np.zeros((32, rank8), dtype=float)
    W8 = np.zeros((32, rank8), dtype=float)
    
    # Left half: columns 0..48
    U8[:, :49] = U4
    for j in range(4):
        for k in range(4):
            V8[j*8+k, :49] = V4[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W8[k*4+i, :49] = W4[k*4+i, :]
    
    # Right half: columns 49..97
    U8[:, 49:] = U4
    for j in range(4):
        for k in range(4):
            V8[j*8+(k+4), 49:] = V4[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W8[(k+4)*4+i, 49:] = W4[k*4+i, :]
    
    # Verify 4x4x8 decomposition
    valid8 = True
    for i in range(4):
        for j in range(4):
            for k in range(8):
                val = np.sum(U8[i*4+j, :] * V8[j*8+k, :] * W8[k*4+i, :])
                if abs(val - 1.0) > 1e-8:
                    valid8 = False
                    break
            if not valid8:
                break
        if not valid8:
            break
    
    if not valid8:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U[ii*m+jj, col] = 1
                    V[jj*p+kk, col] = 1
                    W[kk*n+ii, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U8.tolist(), "V": V8.tolist(), "W": W8.tolist()}
