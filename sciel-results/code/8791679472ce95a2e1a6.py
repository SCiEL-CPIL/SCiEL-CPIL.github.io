"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition of rank n^log2(7) = 4^log2(7) = 49
    n, m, p = DIMENSIONS
    rank = 49
    
    # U: (16, 49), V: (16, 49), W: (16, 49)
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 formulas
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # For 4x4 matrices, we view them as 2x2 blocks of 2x2 submatrices.
    # A = [[A00, A01], [A10, A11]] where each block is 2x2.
    # B = [[B00, B01], [B10, B11]] where each block is 2x2.
    # C = [[C00, C01], [C10, C11]] where each block is 2x2.
    
    # We need to compute each 2x2 block product using Strassen's 7 multiplications.
    # Each multiplication is of 2x2 matrices.
    # A 2x2 matrix multiplication has rank 7 (Strassen) or 4 (schoolbook).
    # We use Strassen for each 2x2 block multiplication.
    
    # Total rank = 7 (Strassen steps) * 7 (rank of 2x2 Strassen) = 49.
    
    # Let's define the 7 Strassen steps for 2x2 matrices.
    # For a 2x2 matrix A' = [[a00, a01], [a10, a11]] and B' = [[b00, b01], [b10, b11]]
    # The 7 products are:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # And the result:
    # c00 = M1 + M4 - M5 - M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M7
    
    # Now, for the 4x4 case, we have 7 such "meta-steps" (M1 to M7 from the 2x2 block level).
    # For each meta-step, we need to compute a product of two 2x2 matrices using Strassen.
    # So for each meta-step s (1..7), we have 7 sub-steps (strassen steps for 2x2).
    # Total = 7 * 7 = 49.
    
    # Let's denote the meta-step as s (0..6) and the sub-step as t (0..6).
    # The global rank index r = s * 7 + t.
    
    # For a given meta-step s, we define linear combinations P_s and Q_s of A and B.
    # Then we compute P_s * Q_s using Strassen (2x2 case).
    
    # Let's define the 7 Strassen steps for 2x2 matrices explicitly.
    # For 2x2 matrices X and Y:
    # Step 0: (x00+x11)(y00+y11) -> contributes to c00, c11
    # Step 1: (x10+x11)y00 -> contributes to c10, c11
    # Step 2: x00(y10-y11) -> contributes to c01, c11
    # Step 3: x11(y01-y00) -> contributes to c00, c10
    # Step 4: (x00+x01)y11 -> contributes to c00, c01
    # Step 5: (x10-x00)(y00+y01) -> contributes to c00, c01, c10
    # Step 6: (x01-x11)(y10+y11) -> contributes to c00, c01, c11
    
    # Let's build the U, V, W matrices.
    
    # First, let's define the 2x2 Strassen decomposition.
    # U_2: (4, 7), V_2: (4, 7), W_2: (4, 7)
    # Indexing: U_2[i*2+j, t], V_2[j*2+k, t], W_2[k*2+i, t]
    
    U_2 = [[0.0] * 7 for _ in range(4)]
    V_2 = [[0.0] * 7 for _ in range(4)]
    W_2 = [[0.0] * 7 for _ in range(4)]
    
    # Strassen 2x2:
    # M1 = (a00+a11)(b00+b11) -> t=0
    U_2[0][0] = 1  # a00
    U_2[3][0] = 1  # a11 (index 3 is i=1,j=1)
    V_2[0][0] = 1  # b00
    V_2[3][0] = 1  # b11 (index 3 is j=1,k=1)
    W_2[0][0] = 1  # c00 (i=0,k=0)
    W_2[3][0] = 1  # c11 (i=1,k=1)
    
    # M2 = (a10+a11)b00 -> t=1
    U_2[2][1] = 1  # a10 (index 2 is i=1,j=0)
    U_2[3][1] = 1  # a11
    V_2[0][1] = 1  # b00
    W_2[2][1] = 1  # c10 (i=1,k=0)
    W_2[3][1] = -1 # c11
    
    # M3 = a00(b10-b11) -> t=2
    U_2[0][2] = 1  # a00
    V_2[2][2] = 1  # b10 (index 2 is j=1,k=0)
    V_2[3][2] = -1 # b11
    W_2[1][2] = 1  # c01 (i=0,k=1)
    W_2[3][2] = 1  # c11
    
    # M4 = a11(b01-b00) -> t=3
    U_2[3][3] = 1  # a11
    V_2[1][3] = 1  # b01 (index 1 is j=0,k=1)
    V_2[0][3] = -1 # b00
    W_2[0][3] = 1  # c00
    W_2[2][3] = 1  # c10
    
    # M5 = (a00+a01)b11 -> t=4
    U_2[0][4] = 1  # a00
    U_2[1][4] = 1  # a01 (index 1 is i=0,j=1)
    V_2[3][4] = 1  # b11
    W_2[0][4] = -1 # c00
    W_2[1][4] = 1  # c01
    
    # M6 = (a10-a00)(b00+b01) -> t=5
    U_2[2][5] = 1  # a10
    U_2[0][5] = -1 # a00
    V_2[0][5] = 1  # b00
    V_2[1][5] = 1  # b01
    W_2[0][5] = -1 # c00
    W_2[1][5] = -1 # c01
    W_2[2][5] = 1  # c10
    
    # M7 = (a01-a11)(b10+b11) -> t=6
    U_2[1][6] = 1  # a01
    U_2[3][6] = -1 # a11
    V_2[2][6] = 1  # b10
    V_2[3][6] = 1  # b11
    W_2[0][6] = -1 # c00
    W_2[1][6] = 1  # c01
    W_2[3][6] = 1  # c11
    
    # Now, for the 4x4 case, we use Strassen at the block level.
    # A = [[A00, A01], [A10, A11]] where each block is 2x2.
    # B = [[B00, B01], [B10, B11]] where each block is 2x2.
    
    # The 7 meta-steps:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # For each meta-step s, we compute P_s * Q_s where P_s and Q_s are 2x2 matrices.
    # Then we use Strassen (2x2) to compute P_s * Q_s.
    
    # Let's define P_s and Q_s as linear combinations of the blocks.
    # For each meta-step s, we have:
    # P_s = alpha_s00 * A00 + alpha_s01 * A01 + alpha_s10 * A10 + alpha_s11 * A11
    # Q_s = beta_s00 * B00 + beta_s01 * B01 + beta_s10 * B10 + beta_s11 * B11
    
    # The coefficients for each meta-step:
    meta_coeffs_A = [
        # s=0: (A00+A11)(B00+B11)
        [[1, 0, 0, 1]],  # A00, A01, A10, A11
        # s=1: (A10+A11)B00
        [[0, 0, 1, 1]],
        # s=2: A00(B10-B11)
        [[1, 0, 0, 0]],
        # s=3: A11(B01-B00)
        [[0, 0, 0, 1]],
        # s=4: (A00+A01)B11
        [[1, 1, 0, 0]],
        # s=5: (A10-A00)(B00+B01)
        [[-1, 0, 1, 0]],
        # s=6: (A01-A11)(B10+B11)
        [[0, 1, 0, -1]]
    ]
    
    meta_coeffs_B = [
        # s=0: (A00+A11)(B00+B11)
        [[1, 0, 0, 1]],  # B00, B01, B10, B11
        # s=1: (A10+A11)B00
        [[1, 0, 0, 0]],
        # s=2: A00(B10-B11)
        [[0, 0, 1, -1]],
        # s=3: A11(B01-B00)
        [[-1, 1, 0, 0]],
        # s=4: (A00+A01)B11
        [[0, 0, 0, 1]],
        # s=5: (A10-A00)(B00+B01)
        [[1, 1, 0, 0]],
        # s=6: (A01-A11)(B10+B11)
        [[0, 0, 1, 1]]
    ]
    
    # Now, for each meta-step s, we compute P_s * Q_s using Strassen (2x2).
    # P_s is a 2x2 matrix, Q_s is a 2x2 matrix.
    # The product C_s = P_s * Q_s is a 2x2 matrix.
    # We use Strassen to compute C_s with rank 7.
    
    # For each meta-step s and each Strassen step t, we have a rank component.
    # The global rank index is r = s * 7 + t.
    
    # For a given s and t, the contribution to U, V, W is:
    # U[i*m+j, r] = sum over blocks of (coeff_A[s][block] * U_2[i*2+j, t])
    # V[j*p+k, r] = sum over blocks of (coeff_B[s][block] * V_2[j*2+k, t])
    # W[k*n+i, r] = W_2[k*2+i, t] (no scaling, since the scaling is in U and V)
    
    # Let's implement this.
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            # U_2 and V_2 are already defined for the 2x2 Strassen steps.
            # For each entry in U_2[i*2+j, t], we add coeff_A[s][block] * U_2[i*2+j, t] to U[i*m+j, r].
            # The blocks are: A00 (i=0,j=0), A01 (i=0,j=1), A10 (i=1,j=0), A11 (i=1,j=1).
            # So for i_block in {0,1} and j_block in {0,1}:
            # The local index in U_2 is (i_block*2 + i_local)*2 + (j_block*2 + j_local)
            # But U_2 is indexed by (i*2+j, t) where i,j are 0 or 1.
            # So for a given t, U_2[i*2+j, t] is non-zero for some i,j.
            # We need to map (i_block, j_block, i_local, j_local) to (i*m+j, r) = (i_block*2 + i_local, j_block*2 + j_local, r).
            
            # Let's iterate over all entries in U_2 for step t.
            for i_local in range(2):
                for j_local in range(2):
                    val_u = U_2[i_local * 2 + j_local][t]
                    if val_u != 0:
                        for i_block in range(2):
                            for j_block in range(2):
                                coeff = meta_coeffs_A[s][0][i_block * 2 + j_block]
                                if coeff != 0:
                                    i_global = i_block * 2 + i_local
                                    j_global = j_block * 2 + j_local
                                    U[i_global * m + j_global][r] += coeff * val_u
            
            for j_local in range(2):
                for k_local in range(2):
                    val_v = V_2[j_local * 2 + k_local][t]
                    if val_v != 0:
                        for j_block in range(2):
                            for k_block in range(2):
                                coeff = meta_coeffs_B[s][0][j_block * 2 + k_block]
                                if coeff != 0:
                                    j_global = j_block * 2 + j_local
                                    k_global = k_block * 2 + k_local
                                    V[j_global * p + k_global][r] += coeff * val_v
            
            for k_local in range(2):
                for i_local in range(2):
                    val_w = W_2[k_local * 2 + i_local][t]
                    if val_w != 0:
                        for k_block in range(2):
                            for i_block in range(2):
                                # W is not scaled by meta coefficients, because the scaling is in U and V.
                                # Wait, is that correct?
                                # Let's re-derive.
                                # C_s = P_s * Q_s = (sum_alpha A_blocks) * (sum_beta B_blocks)
                                # = sum_{alpha, beta} alpha * beta * (A_block * B_block)
                                # But we are using Strassen to compute P_s * Q_s directly.
                                # So P_s is already the sum, and Q_s is already the sum.
                                # So C_s = P_s * Q_s is computed by Strassen.
                                # The Strassen decomposition of P_s * Q_s gives:
                                # C_s = sum_t W_2[:, t] (U_2[:, t] * P_s) (V_2[:, t] * Q_s)
                                # = sum_t W_2[:, t] (sum_{i,j} U_2[i*2+j, t] P_s[i,j]) (sum_{j',k'} V_2[j'*2+k', t] Q_s[j',k'])
                                # = sum_t W_2[:, t] (sum_{i,j} U_2[i*2+j, t] (sum_{alpha} alpha A_block[i,j])) (sum_{j',k'} V_2[j'*2+k', t] (sum_{beta} beta B_block[j',k']))
                                # = sum_t W_2[:, t] (sum_{alpha} alpha (sum_{i,j} U_2[i*2+j, t] A_block[i,j])) (sum_{beta} beta (sum_{j',k'} V_2[j'*2+k', t] B_block[j',k']))
                                # So the coefficients alpha and beta are in U and V, not in W.
                                # So W is not scaled.
                                k_global = k_block * 2 + k_local
                                i_global = i_block * 2 + i_local
                                W[k_global * n + i_global][r] += val_w
    
    # Convert to integer if possible
    for mat in [U, V, W]:
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                if abs(mat[i][j] - round(mat[i][j])) < 1e-9:
                    mat[i][j] = int(round(mat[i][j]))
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
