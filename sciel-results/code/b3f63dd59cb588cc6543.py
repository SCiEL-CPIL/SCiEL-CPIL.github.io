import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition for 4x4x4 and 4x4x1 separately and concatenate.
    
    # Part 1: 4x4x4 (First 4 columns of B)
    # Using Strassen for 4x4x4 (rank 49)
    
    # Strassen 2x2x2 decomposition (rank 7)
    # U (4, 7), V (4, 7), W (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U_in = np.zeros((4, 7), dtype=int)
    V_in = np.zeros((4, 7), dtype=int)
    W_in = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    # A part: A00, A11. B part: B00, B11. C part: C00, C11 (with coeffs)
    # Actually, it's easier to define the block coefficients for Strassen.
    
    # Block coefficients for Strassen 2x2
    # A_blocks: 2x2 matrix of coefficients for A00, A01, A10, A11
    # B_blocks: 2x2 matrix of coefficients for B00, B01, B10, B11
    # C_blocks: 2x2 matrix of coefficients for C00, C01, C10, C11
    
    # M1: (A00+A11)(B00+B11)
    A_b1 = [[1, 0], [0, 1]]
    B_b1 = [[1, 0], [0, 1]]
    C_b1 = [[1, 0], [0, 1]] # M1 contributes to C00 and C11?
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's define the 7 terms properly.
    # Term 0: M1
    # A: A00+A11, B: B00+B11
    # Term 1: M2
    # A: A10+A11, B: B00
    # Term 2: M3
    # A: A00, B: B10-B11
    # Term 3: M4
    # A: A11, B: B01-B00
    # Term 4: M5
    # A: A00+A01, B: B11
    # Term 5: M6
    # A: A10-A00, B: B00+B01
    # Term 6: M7
    # A: A01-A11, B: B10+B11
    
    # C coefficients:
    # C00: M1, M4, -M5, M7
    # C01: M3, M5
    # C10: M2, M4
    # C11: M1, -M2, M3, M6

    A_coeffs = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[-1, 0], [1, 0]], # M6
        [[0, 1], [0, -1]]  # M7
    ]
    B_coeffs = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, -1], [0, 1]], # M3
        [[-1, 1], [0, 0]], # M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 1], [0, 1]]  # M7
    ]
    C_coeffs = [
        [[1, 0], [0, 1]], # M1 -> C00, C11
        [[0, 0], [1, -1]], # M2 -> C10, C11
        [[0, 1], [1, 0]], # M3 -> C01, C11
        [[1, 0], [0, 1]], # M4 -> C00, C10
        [[-1, 1], [0, 0]], # M5 -> C00, C01
        [[0, 0], [1, 1]], # M6 -> C10, C11
        [[1, 0], [0, 0]]  # M7 -> C00
    ]
    # Wait, let's re-verify C coefficients.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1: C00(+1), C11(+1)
    # M2: C10(+1), C11(-1)
    # M3: C01(+1), C11(+1)
    # M4: C00(+1), C10(+1)
    # M5: C00(-1), C01(+1)
    # M6: C10(+1), C11(+1)
    # M7: C00(+1)
    
    # My C_coeffs above:
    # M1: [[1,0],[0,1]] -> C00=1, C11=1. Correct.
    # M2: [[0,0],[1,-1]] -> C10=1, C11=-1. Correct.
    # M3: [[0,1],[1,0]] -> C01=1, C11=1. Correct.
    # M4: [[1,0],[0,1]] -> C00=1, C10=1. Correct.
    # M5: [[-1,1],[0,0]] -> C00=-1, C01=1. Correct.
    # M6: [[0,0],[1,1]] -> C10=1, C11=1. Correct.
    # M7: [[1,0],[0,0]] -> C00=1. Correct.

    # Now, for each term s (0..6), we use the inner Strassen 2x2x2 decomposition.
    # We need U_in, V_in, W_in for 2x2x2 Strassen.
    # Let's generate U_in, V_in, W_in using the same logic but for 2x2.
    # Actually, we can just use the schoolbook for the inner 2x2x2 to simplify code?
    # Rank would be 7 * 8 = 56.
    # Rank 49 is better. Let's stick to Strassen for inner too.
    
    # Inner Strassen 2x2x2
    # We can reuse the A_coeffs, B_coeffs, C_coeffs for the inner level?
    # Yes, the structure is identical.
    
    # We will build the global U, V, W for 4x4x4.
    # U_global (16, 49), V_global (16, 49), W_global (16, 49)
    
    U_g = np.zeros((16, 49), dtype=int)
    V_g = np.zeros((16, 49), dtype=int)
    W_g = np.zeros((16, 49), dtype=int)
    
    # We need the inner U_in, V_in, W_in for 2x2x2 Strassen.
    # Let's compute them once.
    U_in = np.zeros((4, 7), dtype=int)
    V_in = np.zeros((4, 7), dtype=int)
    W_in = np.zeros((4, 7), dtype=int)
    
    # For inner 2x2, indices are i,j,k in 0..1.
    # U_in[i*2+j, t], V_in[j*2+k, t], W_in[k*2+i, t]
    
    # We can use the same A_coeffs etc for the inner level?
    # No, the inner level is just 2x2x2.
    # The coefficients A_coeffs etc define the block structure for 4x4x4.
    # For the inner 2x2x2, we are multiplying 2x2 matrices.
    # So we use Strassen again.
    # So U_in, V_in, W_in are the Strassen decomposition of 2x2x2.
    
    # Let's populate U_in, V_in, W_in using the same logic as above but for size 2.
    # Actually, we can just hardcode the Strassen 2x2 decomposition.
    
    # M1: (A00+A11)(B00+B11)
    # A: 00, 11. B: 00, 11.
    # This is getting complex to nest.
    
    # Alternative: Use Schoolbook for inner 2x2x2.
    # Rank 7 * 8 = 56.
    # Total 56 + 16 = 72.
    # 72 < 80. This is safer and simpler to implement correctly.
    
    # Let's use Schoolbook for inner 2x2x2.
    # U_in (4, 8), V_in (4, 8), W_in (4, 8)
    U_in = np.zeros((4, 8), dtype=int)
    V_in = np.zeros((4, 8), dtype=int)
    W_in = np.zeros((4, 8), dtype=int)
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U_in[i*2+j, col] = 1
                V_in[j*2+k, col] = 1
                W_in[k*2+i, col] = 1
                col += 1
                
    # Now build 4x4x4 using Strassen blocks and Schoolbook inner.
    # Rank = 7 * 8 = 56.
    
    U_g = np.zeros((16, 56), dtype=int)
    V_g = np.zeros((16, 56), dtype=int)
    W_g = np.zeros((16, 56), dtype=int)
    
    col_g = 0
    for s in range(7):
        A_s = A_coeffs[s]
        B_s = B_coeffs[s]
        C_s = C_coeffs[s]
        
        for t in range(8):
            # u_in, v_in, w_in for this t
            for i in range(2):
                for j in range(2):
                    u_val = U_in[i*2+j, t]
                    if u_val != 0:
                        for I in range(2):
                            for J in range(2):
                                a_coeff = A_s[I][J]
                                if a_coeff != 0:
                                    # Global A index: (I*2+i) * 4 + (J*2+j)
                                    idx_a = (I*2+i)*4 + (J*2+j)
                                    U_g[idx_a, col_g] += a_coeff * u_val
                                    
            for j in range(2):
                for k in range(2):
                    v_val = V_in[j*2+k, t]
                    if v_val != 0:
                        for J in range(2):
                            for K in range(2):
                                b_coeff = B_s[J][K]
                                if b_coeff != 0:
                                    # Global B index: (J*2+j) * 4 + (K*2+k)
                                    idx_b = (J*2+j)*4 + (K*2+k)
                                    V_g[idx_b, col_g] += b_coeff * v_val
                                    
            for k in range(2):
                for i in range(2):
                    w_val = W_in[k*2+i, t]
                    if w_val != 0:
                        for K in range(2):
                            for L in range(2):
                                c_coeff = C_s[K][L]
                                if c_coeff != 0:
                                    # Global C index: (K*2+k) * 4 + (L*2+i)
                                    idx_c = (K*2+k)*4 + (L*2+i)
                                    W_g[idx_c, col_g] += c_coeff * w_val
                                    
            col_g += 1
            
    # Part 2: 4x4x1 (Last column of B)
    # Rank 16.
    # U_p2 (16, 16), V_p2 (4, 16), W_p2 (4, 16)
    # V_p2 corresponds to B_{4x1}. Index j*1 + 0 = j.
    # W_p2 corresponds to C_{4x1}. Index k*4 + i = i (since k=0).
    
    U_p2 = np.zeros((16, 16), dtype=int)
    V_p2 = np.zeros((4, 16), dtype=int)
    W_p2 = np.zeros((4, 16), dtype=int)
    
    col_p2 = 0
    # Terms: A11 B1, A12 B2, A21 B1, A22 B2
    # A11: I=0, J=0. B1: J=0, K=0 (rows 0,1).
    # A12: I=0, J=1. B2: J=1, K=1 (rows 2,3).
    # A21: I=1, J=0. B1: J=0, K=0.
    # A22: I=1, J=1. B2: J=1, K=1.
    
    # For each term, we have 4 scalar multiplications.
    # A_{i,j} * B_{j,0} -> C_{i,0}
    
    terms = [
        (0, 0, 0, 0), # A11, B1 -> I=0, J=0
        (0, 1, 1, 1), # A12, B2 -> I=0, J=1
        (1, 0, 0, 0), # A21, B1 -> I=1, J=0
        (1, 1, 1, 1)  # A22, B2 -> I=1, J=1
    ]
    
    for I, J, J_b, K_b in terms:
        # J_b is the block row of B. K_b is not used for B col, but for indexing.
        # B is 4x1. Rows 0,1 are B1. Rows 2,3 are B2.
        # So for B1, global j is 0,1. For B2, global j is 2,3.
        # My term definition:
        # A11 (I=0, J=0) multiplies B1 (rows 0,1).
        # A12 (I=0, J=1) multiplies B2 (rows 2,3).
        # A21 (I=1, J=0) multiplies B1 (rows 0,1).
        # A22 (I=1, J=1) multiplies B2 (rows 2,3).
        
        # In the tuple (I, J, J_b, K_b), J_b is the block index of B.
        # If J_b=0, B rows are 0,1. If J_b=1, B rows are 2,3.
        
        for i in range(2):
            for j in range(2):
                # A index: (I*2+i) * 4 + (J*2+j)
                idx_a = (I*2+i)*4 + (J*2+j)
                # B index: (J_b*2+j) * 1 + 0 = J_b*2+j
                idx_b = J_b*2+j
                # C index: 0 * 4 + (I*2+i) = I*2+i
                idx_c = I*2+i
                
                U_p2[idx_a, col_p2] = 1
                V_p2[idx_b, col_p2] = 1
                W_p2[idx_c, col_p2] = 1
                col_p2 += 1
                
    # Combine
    U = np.hstack([U_g, U_p2])
    V = np.hstack([V_g, V_p2])
    W = np.hstack([W_g, W_p2])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
