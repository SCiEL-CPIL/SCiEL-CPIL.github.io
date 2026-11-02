# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    
    # Rank 85
    rank = 85
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to add a term: U[idx_u] += c_u, V[idx_v] += c_v, W[idx_w] += c_w
    def add_term(u_idx, c_u, v_idx, c_v, w_idx, c_w, col_idx):
        U[u_idx, col_idx] += c_u
        V[v_idx, col_idx] += c_v
        W[w_idx, col_idx] += c_w

    # Part 1: A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    # Using Strassen 4x4x4 (Rank 49)
    
    # Define block indices for A (4x4)
    # A_00: i in 0,1; j in 0,1
    # A_01: i in 0,1; j in 2,3
    # A_10: i in 2,3; j in 0,1
    # A_11: i in 2,3; j in 2,3
    
    # Define block indices for B (4x4)
    # B_00: j in 0,1; k in 0,1
    # B_01: j in 0,1; k in 2,3
    # B_10: j in 2,3; k in 0,1
    # B_11: j in 2,3; k in 2,3
    
    # Strassen Formulas for 4x4 (using 2x2 blocks)
    # M_1 = A_00 + A_11
    # N_1 = B_00 + B_11
    # P_1 = M_1 * N_1
    
    # M_2 = A_10 + A_11
    # N_2 = B_00
    # P_2 = M_2 * N_2
    
    # M_3 = A_00
    # N_3 = B_01 - B_11
    # P_3 = M_3 * N_3
    
    # M_4 = A_11
    # N_4 = B_10 - B_00
    # P_4 = M_4 * N_4
    
    # M_5 = A_00 + A_01
    # N_5 = B_11
    # P_5 = M_5 * N_5
    
    # M_6 = A_10 - A_00
    # N_6 = B_00 + B_01
    # P_6 = M_6 * N_6
    
    # M_7 = A_01 - A_11
    # N_7 = B_10 + B_11
    # P_7 = M_7 * N_7
    
    # Result C_4x4:
    # C_00 = P_1 + P_4 - P_5 + P_7
    # C_01 = P_3 + P_5
    # C_10 = P_2 + P_4
    # C_11 = P_1 - P_2 + P_3 + P_6
    
    # We need to decompose each P_k (2x2 * 2x2) into 7 scalar multiplications.
    # Function to add Strassen decomposition for 2x2 * 2x2
    # Inputs: 
    #   M_coeffs: list of 4 lists of (coeff, a_idx) for M_00, M_01, M_10, M_11
    #   N_coeffs: list of 4 lists of (coeff, b_idx) for N_00, N_01, N_10, N_11
    #   C_coeffs: list of 4 lists of (coeff, c_idx) for C_00, C_01, C_10, C_11
    #   col_start: starting column index
    
    def add_strassen_2x2(M_coeffs, N_coeffs, C_coeffs, col_start):
        # M_coeffs[0] is for M_00, etc.
        # Each entry is a list of (coeff, index)
        
        # Strassen steps:
        # S1 = (M00 + M11) * (N00 + N11)
        # S2 = (M10 + M11) * N00
        # S3 = M00 * (N01 - N11)
        # S4 = M11 * (N10 - N00)
        # S5 = (M00 + M01) * N11
        # S6 = (M10 - M00) * (N00 + N01)
        # S7 = (M01 - M11) * (N10 + N11)
        
        # C00 = S1 + S4 - S5 + S7
        # C01 = S3 + S5
        # C10 = S2 + S4
        # C11 = S1 - S2 + S3 + S6
        
        # We will generate 7 terms.
        # Each term corresponds to one S_k.
        # But S_k is a product of two sums.
        # We need to expand (sum M) * (sum N) into sum of scalar products.
        # Wait, the rank is 7 for 2x2.
        # This means we can represent the 4 outputs using 7 scalar multiplications.
        # The scalar multiplications are of the form u * v.
        # u is a linear combo of M entries, v is a linear combo of N entries.
        
        # Let's define the 7 u and v vectors.
        # u1 = M00 + M11
        # v1 = N00 + N11
        # u2 = M10 + M11
        # v2 = N00
        # u3 = M00
        # v3 = N01 - N11
        # u4 = M11
        # v4 = N10 - N00
        # u5 = M00 + M01
        # v5 = N11
        # u6 = M10 - M00
        # v6 = N00 + N01
        # u7 = M01 - M11
        # v7 = N10 + N11
        
        # We need to map these to the original A and B indices.
        # M_coeffs[i] is a list of (c, idx_a).
        # So u1 is sum of M00 and M11.
        # u1_coeffs = M_coeffs[0] + M_coeffs[3] (summing coefficients for same idx)
        
        def combine_coeffs(list1, list2, sign=1):
            # list1, list2 are lists of (coeff, idx)
            # Returns list of (coeff, idx)
            d = {}
            for c, idx in list1:
                d[idx] = d.get(idx, 0) + c
            for c, idx in list2:
                d[idx] = d.get(idx, 0) + sign * c
            return list(d.items())
        
        def combine_coeffs_4(l0, l1, l2, l3, s0=1, s1=1, s2=1, s3=1):
            d = {}
            for c, idx in l0: d[idx] = d.get(idx, 0) + s0*c
            for c, idx in l1: d[idx] = d.get(idx, 0) + s1*c
            for c, idx in l2: d[idx] = d.get(idx, 0) + s2*c
            for c, idx in l3: d[idx] = d.get(idx, 0) + s3*c
            return list(d.items())
            
        # u1
        u1 = combine_coeffs(M_coeffs[0], M_coeffs[3], 1)
        v1 = combine_coeffs(N_coeffs[0], N_coeffs[3], 1)
        
        # u2
        u2 = combine_coeffs(M_coeffs[2], M_coeffs[3], 1)
        v2 = N_coeffs[0]
        
        # u3
        u3 = M_coeffs[0]
        v3 = combine_coeffs(N_coeffs[1], N_coeffs[3], -1)
        
        # u4
        u4 = M_coeffs[3]
        v4 = combine_coeffs(N_coeffs[2], N_coeffs[0], -1)
        
        # u5
        u5 = combine_coeffs(M_coeffs[0], M_coeffs[1], 1)
        v5 = N_coeffs[3]
        
        # u6
        u6 = combine_coeffs(M_coeffs[2], M_coeffs[0], -1)
        v6 = combine_coeffs(N_coeffs[0], N_coeffs[1], 1)
        
        # u7
        u7 = combine_coeffs(M_coeffs[1], M_coeffs[3], -1)
        v7 = combine_coeffs(N_coeffs[2], N_coeffs[3], 1)
        
        us = [u1, u2, u3, u4, u5, u6, u7]
        vs = [v1, v2, v3, v4, v5, v6, v7]
        
        # C outputs
        # C00 = S1 + S4 - S5 + S7
        # C01 = S3 + S5
        # C10 = S2 + S4
        # C11 = S1 - S2 + S3 + S6
        
        # We add 7 terms to U, V, W.
        # Term k uses u_k and v_k.
        # The contribution to C is determined by the coefficients.
        
        # We need to accumulate the coefficients for C.
        # For each term k, we add to W.
        # W[idx_c, col] += coeff_c
        
        # Let's iterate through the 7 Strassen terms
        for k in range(7):
            col_idx = col_start + k
            # Add u_k to U
            for c, idx in us[k]:
                U[idx, col_idx] += c
            # Add v_k to V
            for c, idx in vs[k]:
                V[idx, col_idx] += c
            
            # Determine W coefficients
            # S1 contributes to C00 (+1), C11 (+1)
            # S2 contributes to C10 (+1), C11 (-1)
            # S3 contributes to C01 (+1), C11 (+1)
            # S4 contributes to C00 (+1), C10 (+1)
            # S5 contributes to C00 (-1), C01 (+1)
            # S6 contributes to C11 (+1)
            # S7 contributes to C00 (+1)
            
            w_coeffs = {0: 0, 1: 0, 2: 0, 3: 0} # C00, C01, C10, C11
            
            if k == 0: # S1
                w_coeffs[0] += 1
                w_coeffs[3] += 1
            elif k == 1: # S2
                w_coeffs[2] += 1
                w_coeffs[3] -= 1
            elif k == 2: # S3
                w_coeffs[1] += 1
                w_coeffs[3] += 1
            elif k == 3: # S4
                w_coeffs[0] += 1
                w_coeffs[2] += 1
            elif k == 4: # S5
                w_coeffs[0] -= 1
                w_coeffs[1] += 1
            elif k == 5: # S6
                w_coeffs[3] += 1
            elif k == 6: # S7
                w_coeffs[0] += 1
            
            for c_idx, c_val in w_coeffs.items():
                if c_val != 0:
                    # C_coeffs[c_idx] is list of (coeff, idx_c)
                    for cc, idx_c in C_coeffs[c_idx]:
                        W[idx_c, col_idx] += c_val * cc
                        
    # Define A block coefficients for 4x4 Strassen
    # A_00: (0,0), (0,1), (1,0), (1,1)
    # A_01: (0,2), (0,3), (1,2), (1,3)
    # A_10: (2,0), (2,1), (3,0), (3,1)
    # A_11: (2,2), (2,3), (3,2), (3,3)
    
    def get_A_block(r1, r2, c1, c2):
        # Returns list of 4 lists of (coeff, idx)
        # idx = i*m + j
        res = [[], [], [], []] # 00, 01, 10, 11
        # 00
        res[0].append((1, r1*m + c1))
        res[0].append((1, r1*m + c1+1))
        res[0].append((1, (r1+1)*m + c1))
        res[0].append((1, (r1+1)*m + c1+1))
        # 01
        res[1].append((1, r1*m + c2))
        res[1].append((1, r1*m + c2+1))
        res[1].append((1, (r1+1)*m + c2))
        res[1].append((1, (r1+1)*m + c2+1))
        # 10
        res[2].append((1, (r1+2)*m + c1))
        res[2].append((1, (r1+2)*m + c1+1))
        res[2].append((1, (r1+3)*m + c1))
        res[2].append((1, (r1+3)*m + c1+1))
        # 11
        res[3].append((1, (r1+2)*m + c2))
        res[3].append((1, (r1+2)*m + c2+1))
        res[3].append((1, (r1+3)*m + c2))
        res[3].append((1, (r1+3)*m + c2+1))
        return res

    def get_B_block(r1, r2, c1, c2):
        # B is m x p. idx = j*p + k
        # r1, r2 are row range for B (j)
        # c1, c2 are col range for B (k)
        res = [[], [], [], []]
        # 00
        res[0].append((1, r1*p + c1))
        res[0].append((1, r1*p + c1+1))
        res[0].append((1, (r1+1)*p + c1))
        res[0].append((1, (r1+1)*p + c1+1))
        # 01
        res[1].append((1, r1*p + c2))
        res[1].append((1, r1*p + c2+1))
        res[1].append((1, (r1+1)*p + c2))
        res[1].append((1, (r1+1)*p + c2+1))
        # 10
        res[2].append((1, (r1+2)*p + c1))
        res[2].append((1, (r1+2)*p + c1+1))
        res[2].append((1, (r1+3)*p + c1))
        res[2].append((1, (r1+3)*p + c1+1))
        # 11
        res[3].append((1, (r1+2)*p + c2))
        res[3].append((1, (r1+2)*p + c2+1))
        res[3].append((1, (r1+3)*p + c2))
        res[3].append((1, (r1+3)*p + c2+1))
        return res

    def get_C_block(r1, r2, c1, c2):
        # C is n x p. idx = k*n + i
        # r1, r2 are row range for C (i)
        # c1, c2 are col range for C (k)
        res = [[], [], [], []]
        # 00
        res[0].append((1, c1*n + r1))
        res[0].append((1, c1*n + r1+1))
        res[0].append((1, (c1+1)*n + r1))
        res[0].append((1, (c1+1)*n + r1+1))
        # 01
        res[1].append((1, c2*n + r1))
        res[1].append((1, c2*n + r1+1))
        res[1].append((1, (c2+1)*n + r1))
        res[1].append((1, (c2+1)*n + r1+1))
        # 10
        res[2].append((1, c1*n + r1+2))
        res[2].append((1, c1*n + r1+3))
        res[2].append((1, (c1+1)*n + r1+2))
        res[2].append((1, (c1+1)*n + r1+3))
        # 11
        res[3].append((1, c2*n + r1+2))
        res[3].append((1, c2*n + r1+3))
        res[3].append((1, (c2+1)*n + r1+2))
        res[3].append((1, (c2+1)*n + r1+3))
        return res

    # A_00..A_11 for A[0:4, 0:4]
    A_blocks = get_A_block(0, 4, 0, 4)
    # B_00..B_11 for B[0:4, 0:4]
    B_blocks = get_B_block(0, 4, 0, 4)
    # C_00..C_11 for C[0:4, 0:4]
    C_blocks = get_C_block(0, 4, 0, 4)
    
    # Construct M and N coefficient lists for the 7 Strassen products
    # M1 = A00 + A11
    M1 = [
        combine_coeffs(A_blocks[0][0], A_blocks[3][0]),
        combine_coeffs(A_blocks[0][1], A_blocks[3][1]),
        combine_coeffs(A_blocks[0][2], A_blocks[3][2]),
        combine_coeffs(A_blocks[0][3], A_blocks[3][3])
    ]
    N1 = [
        combine_coeffs(B_blocks[0][0], B_blocks[3][0]),
        combine_coeffs(B_blocks[0][1], B_blocks[3][1]),
        combine_coeffs(B_blocks[0][2], B_blocks[3][2]),
        combine_coeffs(B_blocks[0][3], B_blocks[3][3])
    ]
    
    # M2 = A10 + A11
    M2 = [
        combine_coeffs(A_blocks[2][0], A_blocks[3][0]),
        combine_coeffs(A_blocks[2][1], A_blocks[3][1]),
        combine_coeffs(A_blocks[2][2], A_blocks[3][2]),
        combine_coeffs(A_blocks[2][3], A_blocks[3][3])
    ]
    N2 = B_blocks[0]
    
    # M3 = A00
    M3 = A_blocks[0]
    # N3 = B01 - B11
    N3 = [
        combine_coeffs(B_blocks[1][0], B_blocks[3][0], -1),
        combine_coeffs(B_blocks[1][1], B_blocks[3][1], -1),
        combine_coeffs(B_blocks[1][2], B_blocks[3][2], -1),
        combine_coeffs(B_blocks[1][3], B_blocks[3][3], -1)
    ]
    
    # M4 = A11
    M4 = A_blocks[3]
    # N4 = B10 - B00
    N4 = [
        combine_coeffs(B_blocks[2][0], B_blocks[0][0], -1),
        combine_coeffs(B_blocks[2][1], B_blocks[0][1], -1),
        combine_coeffs(B_blocks[2][2], B_blocks[0][2], -1),
        combine_coeffs(B_blocks[2][3], B_blocks[0][3], -1)
    ]
    
    # M5 = A00 + A01
    M5 = [
        combine_coeffs(A_blocks[0][0], A_blocks[1][0]),
        combine_coeffs(A_blocks[0][1], A_blocks[1][1]),
        combine_coeffs(A_blocks[0][2], A_blocks[1][2]),
        combine_coeffs(A_blocks[0][3], A_blocks[1][3])
    ]
    N5 = B_blocks[3]
    
    # M6 = A10 - A00
    M6 = [
        combine_coeffs(A_blocks[2][0], A_blocks[0][0], -1),
        combine_coeffs(A_blocks[2][1], A_blocks[0][1], -1),
        combine_coeffs(A_blocks[2][2], A_blocks[0][2], -1),
        combine_coeffs(A_blocks[2][3], A_blocks[0][3], -1)
    ]
    # N6 = B00 + B01
    N6 = [
        combine_coeffs(B_blocks[0][0], B_blocks[1][0]),
        combine_coeffs(B_blocks[0][1], B_blocks[1][1]),
        combine_coeffs(B_blocks[0][2], B_blocks[1][2]),
        combine_coeffs(B_blocks[0][3], B_blocks[1][3])
    ]
    
    # M7 = A01 - A11
    M7 = [
        combine_coeffs(A_blocks[1][0], A_blocks[3][0], -1),
        combine_coeffs(A_blocks[1][1], A_blocks[3][1], -1),
        combine_coeffs(A_blocks[1][2], A_blocks[3][2], -1),
        combine_coeffs(A_blocks[1][3], A_blocks[3][3], -1)
    ]
    # N7 = B10 + B11
    N7 = [
        combine_coeffs(B_blocks[2][0], B_blocks[3][0]),
        combine_coeffs(B_blocks[2][1], B_blocks[3][1]),
        combine_coeffs(B_blocks[2][2], B_blocks[3][2]),
        combine_coeffs(B_blocks[2][3], B_blocks[3][3])
    ]
    
    Ms = [M1, M2, M3, M4, M5, M6, M7]
    Ns = [N1, N2, N3, N4, N5, N6, N7]
    
    for k in range(7):
        add_strassen_2x2(Ms[k], Ns[k], C_blocks, col)
        col += 7
        
    # Part 2: A[0:4, 0:4] * B[0:4, 4:5] -> C[0:4, 4:5]
    # Schoolbook 16 terms
    for i in range(4):
        for j in range(4):
            U[i*m + j, col] = 1
            V[j*p + 4, col] = 1
            W[4*n + i, col] = 1
            col += 1
            
    # Part 3: A[0:4, 4:5] * B[4:5, 0:5] -> C[0:4, 0:5]
    # Schoolbook 20 terms
    for i in range(4):
        for k in range(5):
            U[i*m + 4, col] = 1
            V[4*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
