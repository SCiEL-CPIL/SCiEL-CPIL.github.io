# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 9
    # Rank 49 for 4x4*4x4 (Strassen), Rank 16 for 4x4*4x1 (Schoolbook)
    # Total rank = 49 + 49 + 16 = 114
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 1. Generate 2x2 Strassen decomposition
    def get_2x2_strassen():
        n2, m2, p2 = 2, 2, 2
        r2 = 7
        U2 = [[0] * r2 for _ in range(n2 * m2)]
        V2 = [[0] * r2 for _ in range(m2 * p2)]
        W2 = [[0] * r2 for _ in range(p2 * n2)]
        
        # M1: (A11+A22)(B11+B22)
        # A: [[1,0],[0,1]], B: [[1,0],[0,1]], C: [[1,0],[0,1]]
        U2[0][0] += 1; U2[3][0] += 1
        V2[0][0] += 1; V2[3][0] += 1
        W2[0][0] += 1; W2[3][0] += 1
        
        # M2: (A21+A22)B11
        # A: [[0,0],[1,1]], B: [[1,0],[0,0]], C: [[0,0],[1,-1]]
        U2[2][1] += 1; U2[3][1] += 1
        V2[0][1] += 1
        W2[2][1] += 1; W2[3][1] -= 1
        
        # M3: A11(B12-B22)
        # A: [[1,0],[0,0]], B: [[0,1],[0,-1]], C: [[0,1],[0,1]]
        U2[0][2] += 1
        V2[1][2] += 1; V2[3][2] -= 1
        W2[1][2] += 1; W2[3][2] += 1
        
        # M4: A22(B21-B11)
        # A: [[0,0],[0,1]], B: [[-1,0],[1,0]], C: [[1,0],[1,0]]
        U2[3][3] += 1
        V2[0][3] -= 1; V2[2][3] += 1
        W2[0][3] += 1; W2[2][3] += 1
        
        # M5: (A11+A12)B22
        # A: [[1,1],[0,0]], B: [[0,0],[0,1]], C: [[-1,1],[0,0]]
        U2[0][4] += 1; U2[1][4] += 1
        V2[3][4] += 1
        W2[0][4] -= 1; W2[1][4] += 1
        
        # M6: (A21-A11)(B11+B12)
        # A: [[-1,0],[1,0]], B: [[1,1],[0,0]], C: [[0,0],[0,1]]
        U2[0][5] -= 1; U2[2][5] += 1
        V2[0][5] += 1; V2[1][5] += 1
        W2[3][5] += 1
        
        # M7: (A12-A22)(B21+B22)
        # A: [[0,1],[-1,0]], B: [[0,0],[1,1]], C: [[1,0],[0,0]]
        U2[1][6] += 1; U2[3][6] -= 1
        V2[2][6] += 1; V2[3][6] += 1
        W2[0][6] += 1
        
        return U2, V2, W2

    U2, V2, W2 = get_2x2_strassen()
    
    # 2. Define 4x4 Strassen block combinations
    # A = [[A11, A12], [A21, A22]] (2x2 blocks of 2x2)
    # B = [[B11, B12], [B21, B22]]
    
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # We need to map these block coefficients to the global 4x4 indices.
    # Global A index: i*4+j. Block (bi, bj) covers i in [2*bi, 2*bi+1], j in [2*bj, 2*bj+1].
    
    # For each block multiplication r_block (0..6), we have coefficients for A_blocks and B_blocks.
    # Then we use the 2x2 Strassen U2, V2, W2 to expand each block multiplication into 7 scalar multiplications.
    
    # Structure:
    # For r_block in 0..6:
    #   Get A_block_coeffs (2x2 matrix of coeffs for A11..A22)
    #   Get B_block_coeffs (2x2 matrix of coeffs for B11..B22)
    #   For r_scalar in 0..6:
    #       r_global = r_block * 7 + r_scalar
    #       # The scalar multiplication is:
    #       # (Sum_{bi,bj} A_coeff[bi][bj] * A_block[bi][bj]) * (Sum_{bk,bl} B_coeff[bk][bl] * B_block[bk][bl])
    #       # This is equivalent to summing over all block pairs.
    #       # But we can just add the contributions.
    #       
    #       # Actually, it's simpler:
    #       # The term is U2[i_l, j_l][r_scalar] * V2[j_l, k_l][r_scalar] * W2[k_l, i_l][r_scalar]
    #       # where i_l, j_l, k_l are local 2x2 indices.
    #       # The global indices are:
    #       # i_global = 2*bi + i_l
    #       # j_global = 2*bj + j_l
    #       # k_global = 2*bk + k_l
    #       # The coefficient is A_coeff[bi][bj] * B_coeff[bk][bl] * ... wait.
    #       
    #       # The linear form for A is L_A = Sum_{bi,bj} A_coeff[bi][bj] A_{bi,bj}
    #       # The linear form for B is L_B = Sum_{bk,bl} B_coeff[bk][bl] B_{bk,bl}
    #       # The product L_A L_B = Sum_{bi,bj,bk,bl} A_coeff[bi][bj] B_coeff[bk][bl] A_{bi,bj} B_{bk,bl}
    #       # The 2x2 Strassen decomposition of A_{bi,bj} B_{bk,bl} is NOT simply U2...
    #       # Because U2 decomposes the product of TWO 2x2 matrices.
    #       # Here we have a sum of products of 2x2 matrices.
    #       # By linearity, we can just add the decompositions.
    #       
    #       # So for each (bi, bj, bk, bl) where A_coeff and B_coeff are non-zero:
    #       # We add the 2x2 Strassen decomposition of A_{bi,bj} * B_{bk,bl} scaled by the coeffs.
    
    # Let's define the block coeffs for the 7 Strassen steps.
    # A_coeffs[step][bi][bj]
    # B_coeffs[step][bk][bl]
    
    A_coeffs = [
        [[1,0],[0,1]], # M1: A11+A22
        [[0,0],[1,1]], # M2: A21+A22
        [[1,0],[0,0]], # M3: A11
        [[0,0],[0,1]], # M4: A22
        [[1,1],[0,0]], # M5: A11+A12
        [[-1,0],[1,0]],# M6: A21-A11
        [[0,1],[-1,0]] # M7: A12-A22
    ]
    
    B_coeffs = [
        [[1,0],[0,1]], # M1: B11+B22
        [[1,0],[0,0]], # M2: B11
        [[0,1],[0,-1]],# M3: B12-B22
        [[-1,0],[1,0]],# M4: B21-B11
        [[0,0],[0,1]], # M5: B22
        [[1,1],[0,0]], # M6: B11+B12
        [[0,0],[1,1]]  # M7: B21+B22
    ]
    
    # W_coeffs for the 4x4 result
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to add the W contributions from the 2x2 Strassen to the global W.
    # The 2x2 Strassen W2 gives the local C_{i_l, k_l}.
    # The global C_{i_g, k_g} gets contributions from the block C_{bi, bk} scaled by W_block_coeff.
    # But the W_block_coeff is determined by the Strassen formula for 4x4.
    # Actually, the W2 from the 2x2 Strassen gives the result of the block product.
    # We just need to map the local block result to the global result using the 4x4 Strassen W coefficients.
    
    # Let's do it in two steps:
    # 1. Compute the 49 scalar multiplications (U, V) for the 4x4 * 4x4 case.
    # 2. Compute the W matrix for the 4x4 * 4x4 case.
    
    # Step 1: U and V
    # For each step s (0..6):
    #   For each bi, bj where A_coeffs[s][bi][bj] != 0:
    #     For each bk, bl where B_coeffs[s][bk][bl] != 0:
    #       Scale = A_coeffs[s][bi][bj] * B_coeffs[s][bk][bl]
    #       For each r_scalar (0..6):
    #         r_global = s * 7 + r_scalar
    #         # Add U2, V2 contributions for the block (bi,bj) * (bk,bl)
    #         # U2[i_l, j_l][r_scalar] contributes to U[i_g, j_g]
    #         # i_g = 2*bi + i_l, j_g = 2*bj + j_l
    #         # V2[j_l, k_l][r_scalar] contributes to V[j_g, k_g]
    #         # j_g = 2*bj + j_l, k_g = 2*bk + k_l
            
    for s in range(7):
        for bi in range(2):
            for bj in range(2):
                ca = A_coeffs[s][bi][bj]
                if ca == 0: continue
                for bk in range(2):
                    for bl in range(2):
                        cb = B_coeffs[s][bk][bl]
                        if cb == 0: continue
                        scale = ca * cb
                        for r_scalar in range(7):
                            r_global = s * 7 + r_scalar
                            for i_l in range(2):
                                for j_l in range(2):
                                    u_val = U2[i_l*2+j_l][r_scalar]
                                    if u_val != 0:
                                        i_g = 2*bi + i_l
                                        j_g = 2*bj + j_l
                                        U[i_g*4+j_g][r_global] += u_val * scale
                                        
                            for j_l in range(2):
                                for k_l in range(2):
                                    v_val = V2[j_l*2+k_l][r_scalar]
                                    if v_val != 0:
                                        j_g = 2*bj + j_l
                                        k_g = 2*bk + k_l
                                        V[j_g*4+k_l][r_global] += v_val * scale # Wait, V index is j*4+k.
                                        # V[j_g*4+k_g]
                                        
    # Step 2: W
    # W_global[k_g, i_g]
    # C_global = Sum_s W_block_coeff[s][bk, bi] * C_block[s][bk, bi]
    # C_block[s][bk, bi] is the result of the s-th block multiplication.
    # The s-th block multiplication result is computed by the 2x2 Strassen W2.
    # C_block[s][k_l, i_l] = Sum_{r_scalar} W2[k_l*2+i_l][r_scalar] * M_{s, r_scalar}
    # So W_global[k_g, i_g] gets contribution from W2[k_l, i_l][r_scalar] * W_block_coeff[s][bk, bi]
    # where k_g = 2*bk + k_l, i_g = 2*bi + i_l.
    
    # W_block_coeffs for 4x4 Strassen:
    # C11 (bi=0, bk=0): M1(1) + M4(1) - M5(1) + M7(1)
    # C12 (bi=0, bk=1): M3(1) + M5(1)
    # C21 (bi=1, bk=0): M2(1) + M4(1)
    # C22 (bi=1, bk=1): M1(1) - M2(1) + M3(1) + M6(1)
    
    W_block_coeffs = [
        [[0,0],[0,0]], # Placeholder
        [[0,0],[0,0]],
        [[0,0],[0,0]],
        [[0,0],[0,0]],
        [[0,0],[0,0]],
        [[0,0],[0,0]],
        [[0,0],[0,0]]
    ]
    # s=0 (M1): C11+=1, C22+=1
    W_block_coeffs[0][0][0] += 1
    W_block_coeffs[0][1][1] += 1
    # s=1 (M2): C21+=1, C22-=1
    W_block_coeffs[1][0][1] += 1
    W_block_coeffs[1][1][1] -= 1
    # s=2 (M3): C12+=1, C22+=1
    W_block_coeffs[2][1][0] += 1
    W_block_coeffs[2][1][1] += 1
    # s=3 (M4): C11+=1, C21+=1
    W_block_coeffs[3][0][0] += 1
    W_block_coeffs[3][0][1] += 1
    # s=4 (M5): C11-=1, C12+=1
    W_block_coeffs[4][0][0] -= 1
    W_block_coeffs[4][1][0] += 1
    # s=5 (M6): C22+=1
    W_block_coeffs[5][1][1] += 1
    # s=6 (M7): C11+=1
    W_block_coeffs[6][0][0] += 1
    
    for s in range(7):
        for bk in range(2):
            for bi in range(2):
                cw = W_block_coeffs[s][bk][bi]
                if cw == 0: continue
                for r_scalar in range(7):
                    r_global = s * 7 + r_scalar
                    for k_l in range(2):
                        for i_l in range(2):
                            w_val = W2[k_l*2+i_l][r_scalar]
                            if w_val != 0:
                                k_g = 2*bk + k_l
                                i_g = 2*bi + i_l
                                W[k_g*4+i_g][r_global] += w_val * cw

    # 3. Handle the 4x1 part (last column of the 9)
    # A is 4x4, B is 4x1.
    # C is 4x1.
    # Schoolbook: C_i = Sum_j A_{ij} B_j.
    # Rank 16.
    # Global indices:
    # U[i*4+j], V[j*4+k], W[k*4+i]
    # Here k is always 0 (since B is 4x1, only column 0).
    # But in the full 4x4*4x9, the 4x1 part corresponds to the 9th column (index 8).
    # So k_global = 8.
    # The B matrix for this part is the 9th column of the 4x9 matrix.
    # V index: j*9 + 8.
    # W index: 8*4 + i.
    
    offset = 98 # 49 + 49
    for i in range(4):
        for j in range(4):
            r = offset + i*4 + j
            U[i*4+j][r] = 1
            V[j*4+8][r] = 1
            W[8*4+i][r] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
