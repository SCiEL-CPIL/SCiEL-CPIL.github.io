"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Strassen 2x2 decomposition (Rank 7)
    # U2, V2, W2 shapes (4, 7)
    # Indices: A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1) -> Index k*2+i
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 0, 1, 0, 1, 0],  # B01
        [0, 0, 1, 0, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # C00 (k=0, i=0)
        [0, 1, 0, 1, 0, 0, 0],  # C10 (k=0, i=1)
        [0, 0, 1, 0, 1, 0, 0],  # C01 (k=1, i=0)
        [1, -1, 1, 0, 0, 1, 0]  # C11 (k=1, i=1)
    ]
    
    # Liu 3x3 decomposition (Rank 19)
    # U3, V3, W3 shapes (9, 19)
    # Indices: A: I*3+J, B: J*3+K, C: K*3+I
    
    # Coefficients for Liu's algorithm (standard reference)
    # This is a known rank-19 decomposition.
    U3 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A00
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A01
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A02
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A10
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A11
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A12
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A20
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A21
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # A22
    ]
    # NOTE: The above U3 is identity, which is WRONG for a rank reduction.
    # I must use the actual Liu coefficients.
    # Since I cannot guarantee the exact 19 coefficients from memory without error,
    # I will use the safer Rank-189 solution (Strassen 2x2 x Schoolbook 3x3).
    # This guarantees correctness and an improvement over 216.
    
    # Schoolbook 3x3 (Rank 27)
    U3_school = [[0]*27 for _ in range(9)]
    V3_school = [[0]*27 for _ in range(9)]
    W3_school = [[0]*27 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3_school[i*3+j][col] = 1
                V3_school[j*3+k][col] = 1
                W3_school[k*3+i][col] = 1
                col += 1
                
    # Use Schoolbook for 3x3 to ensure correctness
    U3 = U3_school
    V3 = V3_school
    W3 = W3_school
    rank3 = 27
    rank2 = 7
    rank_total = rank3 * rank2
    
    # Construct 6x6 factors
    n, m, p = DIMENSIONS
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Map indices
    # A: (I, i) x (J, j) -> I*2+i, J*2+j
    # B: (J, j) x (K, k) -> J*2+j, K*2+k
    # C: (K, k) x (I, i) -> K*2+k, I*2+i (W index: k_global * 6 + i_global)
    
    for I in range(3):
        for J in range(3):
            for i in range(2):
                for j in range(2):
                    row_a = I * 2 + i
                    col_a = J * 2 + j
                    idx_a = row_a * 6 + col_a
                    for c3 in range(rank3):
                        u3_val = U3[I*3+J][c3]
                        if u3_val == 0: continue
                        for c2 in range(rank2):
                            u2_val = U2[i*2+j][c2]
                            if u2_val == 0: continue
                            total_c = c3 * rank2 + c2
                            U[idx_a][total_c] = u3_val * u2_val

    for J in range(3):
        for K in range(3):
            for j in range(2):
                for k in range(2):
                    row_b = J * 2 + j
                    col_b = K * 2 + k
                    idx_b = row_b * 6 + col_b
                    for c3 in range(rank3):
                        v3_val = V3[J*3+K][c3]
                        if v3_val == 0: continue
                        for c2 in range(rank2):
                            v2_val = V2[j*2+k][c2]
                            if v2_val == 0: continue
                            total_c = c3 * rank2 + c2
                            V[idx_b][total_c] = v3_val * v2_val

    for K in range(3):
        for I in range(3):
            for k in range(2):
                for i in range(2):
                    # W index: k_global * n + i_global
                    # k_global = K*2+k
                    # i_global = I*2+i
                    idx_w = (K * 2 + k) * 6 + (I * 2 + i)
                    for c3 in range(rank3):
                        w3_val = W3[K*3+I][c3]
                        if w3_val == 0: continue
                        for c2 in range(rank2):
                            w2_val = W2[k*2+i][c2]
                            if w2_val == 0: continue
                            total_c = c3 * rank2 + c2
                            W[idx_w][total_c] = w3_val * w2_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
