import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 28
    rank = 28
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # We will fill U, V, W column by column for each of the 28 terms.
    # Term index r corresponds to (k, j, c) where:
    # k is 0..6 (Strassen product index 1..7)
    # j is 0..1 (row of L_k / row of R_k)
    # c is 0..1 (col of R_k)
    
    # Strassen definitions for 2x2 block matrices
    # A_block = [[A00, A01], [A10, A11]]
    # B_block = [[B00, B01], [B10, B11]]
    
    # A_block entries are 1x2 vectors.
    # A00 = A[0, 0:2] -> indices (0,0), (0,1)
    # A01 = A[0, 2:4] -> indices (0,2), (0,3)
    # A10 = A[1, 0:2] -> indices (1,0), (1,1)
    # A11 = A[1, 2:4] -> indices (1,2), (1,3)
    
    # B_block entries are 2x2 matrices.
    # B00 = B[0:2, 0:2] -> indices (0,0)..(1,1)
    # B01 = B[0:2, 2:4] -> indices (0,2)..(1,3)
    # B10 = B[2:4, 0:2] -> indices (2,0)..(3,1)
    # B11 = B[2:4, 2:4] -> indices (2,2)..(3,3)
    
    # Define linear combinations for L_k (1x2 vectors)
    # L_k is a linear combo of A_block entries.
    # A_block entries are 1x2. So L_k is 1x2.
    # L_k[j] is the coefficient for the j-th element of the 1x2 vector?
    # No, L_k is the whole 1x2 vector.
    # L_k = sum_{ab} coeff_{k,ab} * A_ab
    # Where A_ab is a 1x2 vector.
    
    # Define linear combinations for R_k (2x2 matrices)
    # R_k = sum_{bc} coeff_{k,bc} * B_bc
    # Where B_bc is a 2x2 matrix.
    
    # The product M_k = L_k * R_k is a 1x2 vector.
    # M_k[c] = sum_{j=0}^1 L_k[j] * R_k[j, c]
    
    # We decompose M_k[c] into 2 scalar multiplications:
    # M_k[c] = L_k[0]*R_k[0,c] + L_k[1]*R_k[1,c]
    
    # So for each k, c, j, we have a scalar multiplication:
    # term = L_k[j] * R_k[j,c]
    
    # L_k[j] is a scalar linear combo of A's.
    # R_k[j,c] is a scalar linear combo of B's.
    
    # Let's define the coefficients for L_k and R_k.
    # Indices for A_block: 00, 01, 10, 11
    # Indices for B_block: 00, 01, 10, 11
    
    # Strassen products:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Coeffs for L_k (A side)
    # L_k is a 1x2 vector.
    # Since A_ab are 1x2 vectors, L_k is just a linear combo of these vectors.
    # L_k = c00*A00 + c01*A01 + c10*A10 + c11*A11
    # L_k[j] = c00*A00[j] + c01*A01[j] + c10*A10[j] + c11*A11[j]
    
    # Coeffs for R_k (B side)
    # R_k is a 2x2 matrix.
    # R_k = d00*B00 + d01*B01 + d10*B10 + d11*B11
    # R_k[j,c] = d00*B00[j,c] + d01*B01[j,c] + d10*B10[j,c] + d11*B11[j,c]
    
    # Scalar multiplication: L_k[j] * R_k[j,c]
    # = (sum_ab c_k[ab] A_ab[j]) * (sum_bc d_k[bc] B_bc[j,c])
    # This is a bilinear form.
    # The rank-1 decomposition term corresponds to fixing specific A and B elements.
    # Actually, we can just expand the product into sums of scalar products of A elements and B elements.
    
    # Let's iterate through k, j, c.
    
    # Mapping from block index to actual indices
    # A_block[ab] corresponds to A[ab[0], ab[1]*2 : ab[1]*2+2]
    # A00: i=0, j=0,1
    # A01: i=0, j=2,3
    # A10: i=1, j=0,1
    # A11: i=1, j=2,3
    
    # B_block[bc] corresponds to B[bc[0]*2 : bc[0]*2+2, bc[1]*2 : bc[1]*2+2]
    # B00: j=0,1, k=0,1
    # B01: j=0,1, k=2,3
    # B10: j=2,3, k=0,1
    # B11: j=2,3, k=2,3
    
    # Output C_block[ab] corresponds to C[ab[0], ab[1]*2 : ab[1]*2+2]
    # C00: i=0, k=0,1
    # C01: i=0, k=2,3
    # C10: i=1, k=0,1
    # C11: i=1, k=2,3
    
    # Strassen coefficients
    # k, A_coeffs {00,01,10,11}, B_coeffs {00,01,10,11}
    strassen = [
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}), # M1
        ({'10': 1, '11': 1}, {'00': 1}),           # M2
        ({'00': 1}, {'01': 1, '11': -1}),          # M3
        ({'11': 1}, {'10': 1, '00': -1}),          # M4
        ({'00': 1, '01': 1}, {'11': 1}),           # M5
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}), # M6
        ({'01': 1, '11': -1}, {'10': 1, '11': 1})  # M7
    ]
    
    # C_block combination
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M2
    
    # For each M_k, it contributes to C_block entries with certain signs.
    # M_k is a 1x2 vector.
    # M_k[c] contributes to C_block[?][c]?
    # No, M_k is a 1x2 vector.
    # M1 contributes to C00 and C11.
    # M1[0] -> C00[0] and C11[0]
    # M1[1] -> C00[1] and C11[1]
    
    # Let's map M_k index to C_block targets and signs.
    # M1: C00 (+1), C11 (+1)
    # M2: C10 (+1), C11 (-1)
    # M3: C01 (+1), C11 (-1)
    # M4: C10 (+1), C00 (-1)
    # M5: C11 (+1)
    # M6: C00 (+1)
    # M7: C01 (+1), C00 (-1)
    
    targets = {
        0: [('00', 1), ('11', 1)],   # M1
        1: [('10', 1), ('11', -1)],  # M2
        2: [('01', 1), ('11', -1)],  # M3
        3: [('10', 1), ('00', -1)],  # M4
        4: [('11', 1)],              # M5
        5: [('00', 1)],              # M6
        6: [('01', 1), ('00', -1)]   # M7
    }
    
    r = 0
    for k in range(7):
        A_coeffs, B_coeffs = strassen[k]
        
        # A_coeffs maps block index to coefficient
        # B_coeffs maps block index to coefficient
        
        # For each j (0,1) and c (0,1)
        for j in range(2):
            for c in range(2):
                # This term is L_k[j] * R_k[j,c]
                
                # L_k[j] = sum_{ab} A_coeffs[ab] * A_ab[j]
                # R_k[j,c] = sum_{bc} B_coeffs[bc] * B_bc[j,c]
                
                # The product is sum_{ab, bc} A_coeffs[ab] B_coeffs[bc] A_ab[j] B_bc[j,c]
                # Each pair (ab, bc) is a scalar multiplication of A_ab[j] and B_bc[j,c].
                # We need to add a rank-1 term for each non-zero coefficient.
                
                for ab_str, ca in A_coeffs.items():
                    for bc_str, cb in B_coeffs.items():
                        coeff = ca * cb
                        if coeff == 0:
                            continue
                        
                        # Identify A_ab[j] index
                        # ab_str is '00', '01', etc.
                        ai = int(ab_str[0])
                        aj_block = int(ab_str[1])
                        aj = aj_block * 2 + j
                        
                        # U index: i * m + j
                        u_idx = ai * m + aj
                        
                        # Identify B_bc[j,c] index
                        # bc_str is '00', '01', etc.
                        bj_block = int(bc_str[0])
                        bj = bj_block * 2 + j
                        bk_block = int(bc_str[1])
                        bk = bk_block * 2 + c
                        
                        # V index: j * p + k
                        v_idx = bj * p + bk
                        
                        # Determine W index and sign
                        # This term contributes to C_block targets of M_k.
                        # The component is c (0 or 1).
                        # C_block[ab][c] corresponds to C[ai, ak]
                        # where ai = int(ab[0]), ak = int(ab[1])*2 + c
                        
                        for target_ab, sign in targets[k]:
                            ci = int(target_ab[0])
                            ck_block = int(target_ab[1])
                            ck = ck_block * 2 + c
                            
                            # W index: k * n + i
                            w_idx = ck * n + ci
                            
                            # Add to U, V, W
                            U[u_idx, r] += coeff
                            V[v_idx, r] += coeff
                            W[w_idx, r] += coeff * sign
                            
                        r += 1
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
