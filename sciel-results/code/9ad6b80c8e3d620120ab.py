import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x6 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    Returns {'U', 'V', 'W'} with shapes (16, R), (24, R), (24, R).
    Score is the rank R (lower is better).
    """
    n, m, p = 4, 4, 6
    
    # 1. Strassen decomposition for 2x2 x 2x2 (Rank 7)
    # U_s, V_s, W_s for M_{2,2,2}
    # Indices: U[0..3], V[0..3], W[0..3]
    # M[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # Here i is row of A, j is col of A/row of B, k is col of B.
    # Wait, standard tensor indices:
    # A is 2x2, B is 2x2, C is 2x2.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T[i,j,k] = 1.
    # U[i,j, r]? No, U is (n*m, R).
    # For 2x2, n=2, m=2. U is (4, 7).
    # Row index of U: i*m + j.
    # Row index of V: j*p + k. For 2x2, p=2. V is (4, 7).
    # Row index of W: k*n + i. W is (4, 7).
    
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # Strassen terms
    # s=0: (A00+A11)(B00+B11) -> C00 +=, C01 +=, C10 +=, C11 += (with signs)
    # Let's use the standard Strassen formulas:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U part (A factors)
    # A00 is at U[0, :], A01 at U[1, :], A10 at U[2, :], A11 at U[3, :]
    # M1: A00+A11 -> U[0]+=1, U[3]+=1
    U_222[0, 0] += 1; U_222[3, 0] += 1
    # M2: A10+A11 -> U[2]+=1, U[3]+=1
    U_222[2, 1] += 1; U_222[3, 1] += 1
    # M3: A00 -> U[0]+=1
    U_222[0, 2] += 1
    # M4: A11 -> U[3]+=1
    U_222[3, 3] += 1
    # M5: A00+A01 -> U[0]+=1, U[1]+=1
    U_222[0, 4] += 1; U_222[1, 4] += 1
    # M6: A10-A00 -> U[2]+=1, U[0]+=-1
    U_222[2, 5] += 1; U_222[0, 5] += -1
    # M7: A01-A11 -> U[1]+=1, U[3]+=-1
    U_222[1, 6] += 1; U_222[3, 6] += -1
    
    # V part (B factors)
    # B00 is at V[0, :], B01 at V[1, :], B10 at V[2, :], B11 at V[3, :]
    # M1: B00+B11 -> V[0]+=1, V[3]+=1
    V_222[0, 0] += 1; V_222[3, 0] += 1
    # M2: B00 -> V[0]+=1
    V_222[0, 1] += 1
    # M3: B01-B11 -> V[1]+=1, V[3]+=-1
    V_222[1, 2] += 1; V_222[3, 2] += -1
    # M4: B10-B00 -> V[2]+=1, V[0]+=-1
    V_222[2, 3] += 1; V_222[0, 3] += -1
    # M5: B11 -> V[3]+=1
    V_222[3, 4] += 1
    # M6: B00+B01 -> V[0]+=1, V[1]+=1
    V_222[0, 5] += 1; V_222[1, 5] += 1
    # M7: B10+B11 -> V[2]+=1, V[3]+=1
    V_222[2, 6] += 1; V_222[3, 6] += 1
    
    # W part (C factors)
    # C00 is at W[0, :], C01 at W[1, :], C10 at W[2, :], C11 at W[3, :]
    # C00 = M1 + M4 - M5 + M7
    W_222[0, 0] += 1; W_222[0, 3] += 1; W_222[0, 4] += -1; W_222[0, 6] += 1
    # C01 = M3 + M5
    W_222[1, 2] += 1; W_222[1, 4] += 1
    # C10 = M2 + M4
    W_222[2, 1] += 1; W_222[2, 3] += 1
    # C11 = M1 - M2 + M3 + M6
    W_222[3, 0] += 1; W_222[3, 1] += -1; W_222[3, 2] += 1; W_222[3, 5] += 1

    # 2. Decomposition for 2x2 x 2x3 (Rank 11)
    # M_{2,2,3}. A is 2x2, B is 2x3, C is 2x3.
    # U_223 is (4, 11), V_223 is (6, 11), W_223 is (6, 11).
    # Split B into B1 (2x2) and B2 (2x1).
    # C1 = A * B1 (2x2), C2 = A * B2 (2x1).
    # Use Strassen for C1 (7 terms) and Schoolbook for C2 (4 terms).
    
    U_223 = np.zeros((4, 11))
    V_223 = np.zeros((6, 11))
    W_223 = np.zeros((6, 11))
    
    # Part 1: Strassen for 2x2 x 2x2 (columns 0,1 of B, 0,1 of C)
    # Copy U_222, V_222, W_222 into the first 7 columns of U_223, V_223, W_223.
    # V_222 indices 0..3 correspond to B00, B01, B10, B11.
    # In V_223, B00 is at 0, B01 at 1, B10 at 2, B11 at 3.
    # So V_223[0:4, 0:7] = V_222.
    # W_222 indices 0..3 correspond to C00, C01, C10, C11.
    # In W_223, C00 is at 0, C01 at 1, C10 at 2, C11 at 3.
    # So W_223[0:4, 0:7] = W_222.
    
    U_223[:, 0:7] = U_222
    V_223[0:4, 0:7] = V_222
    W_223[0:4, 0:7] = W_222
    
    # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of B, 2,3 of C)
    # B2 is 2x1. Let B2 = [B02, B12].
    # C2 is 2x1. C2 = [C02, C12].
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    # 4 multiplications.
    
    # Term 7: A00 * B02 -> C02
    U_223[0, 7] = 1
    V_223[4, 7] = 1  # B02 is at index 4 in V_223 (row 0, col 2 of B)
    W_223[4, 7] = 1  # C02 is at index 4 in W_223 (row 0, col 2 of C)
    
    # Term 8: A01 * B12 -> C02
    U_223[1, 8] = 1
    V_223[5, 8] = 1  # B12 is at index 5 in V_223 (row 1, col 2 of B)
    W_223[4, 8] = 1  # C02
    
    # Term 9: A10 * B02 -> C12
    U_223[2, 9] = 1
    V_223[4, 9] = 1  # B02
    W_223[5, 9] = 1  # C12 is at index 5 in W_223 (row 1, col 2 of C)
    
    # Term 10: A11 * B12 -> C12
    U_223[3, 10] = 1
    V_223[5, 10] = 1  # B12
    W_223[5, 10] = 1  # C12

    # 3. Kronecker Product
    # M_{4,4,6} = M_{2,2,2} \otimes M_{2,2,3}
    # Rank = 7 * 11 = 77.
    # U_446 is (16, 77), V_446 is (24, 77), W_446 is (24, 77).
    
    R = 77
    U_446 = np.zeros((16, R))
    V_446 = np.zeros((24, R))
    W_446 = np.zeros((24, R))
    
    for r1 in range(7):
        for r2 in range(11):
            r = r1 * 11 + r2
            
            # U part
            # U_222[i1, r1] * U_223[i2, r2]
            # Index in U_446: (i1*2 + i2) * 4 + (j1*2 + j2)?
            # No, U_446 row index is i*m + j.
            # i = i1*2 + i2, j = j1*2 + j2.
            # So index = (i1*2 + i2)*4 + (j1*2 + j2).
            for i1 in range(2):
                for j1 in range(2):
                    u1_val = U_222[i1*2 + j1, r1]
                    if u1_val == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2_val = U_223[i2*2 + j2, r2]
                            if u2_val == 0: continue
                            u_idx = (i1*2 + i2)*4 + (j1*2 + j2)
                            U_446[u_idx, r] += u1_val * u2_val
            
            # V part
            # V_222[j1, r1] * V_223[j2, r2]
            # V_446 row index is j*p + k.
            # j = j1*2 + j2, k = k1*3 + k2?
            # No, B is 4x6.
            # B is 2x2 blocks of 2x3.
            # j = j1*2 + j2.
            # k = k1*3 + k2.
            # Index = (j1*2 + j2)*6 + (k1*3 + k2).
            for j1 in range(2):
                v1_val = V_222[j1, r1] # Note: V_222 row index is j*p+k? No, for 2x2, V row is j*2+k.
                # Wait, V_222 is (4,7). Row index j*2+k.
                # So j1 is row of B1.
                # But V_222 represents B which is 2x2.
                # In the Kronecker product, the "middle" dimension m=4.
                # B is 4x6.
                # B is composed of 2x2 blocks of 2x3.
                # So j = j1*2 + j2.
                # k = k1*3 + k2.
                # V_222 index is j1*2 + k1? No.
                # V_222 is for M_{2,2,2}. B is 2x2.
                # V_222 row index is j*2 + k.
                # So V_222[j1, r1] is not correct.
                # V_222 has 4 rows.
                # The Kronecker product logic:
                # V_446[j, k, r] = sum V_222[j1, k1, r1] V_223[j2, k2, r2]
                # where j = j1*2 + j2, k = k1*3 + k2.
                # V_222 row index is j1*2 + k1.
                # V_223 row index is j2*3 + k2.
                
                for k1 in range(2):
                    v1_idx = j1*2 + k1
                    v1_val = V_222[v1_idx, r1]
                    if v1_val == 0: continue
                    for j2 in range(2):
                        for k2 in range(3):
                            v2_idx = j2*3 + k2
                            v2_val = V_223[v2_idx, r2]
                            if v2_val == 0: continue
                            v_idx = (j1*2 + j2)*6 + (k1*3 + k2)
                            V_446[v_idx, r] += v1_val * v2_val

            # W part
            # W_222[k1, i1, r1] * W_223[k2, i2, r2]
            # W_446 row index is k*n + i.
            # k = k1*3 + k2, i = i1*2 + i2.
            # Index = (k1*3 + k2)*4 + (i1*2 + i2).
            # W_222 row index is k1*2 + i1.
            # W_223 row index is k2*2 + i2.
            
            for k1 in range(2):
                for i1 in range(2):
                    w1_idx = k1*2 + i1
                    w1_val = W_222[w1_idx, r1]
                    if w1_val == 0: continue
                    for k2 in range(3):
                        for i2 in range(2):
                            w2_idx = k2*2 + i2
                            w2_val = W_223[w2_idx, r2]
                            if w2_val == 0: continue
                            w_idx = (k1*3 + k2)*4 + (i1*2 + i2)
                            W_446[w_idx, r] += w1_val * w2_val

    # Convert to lists
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
    
    return {"U": to_list(U_446), "V": to_list(V_446), "W": to_list(W_446)}
