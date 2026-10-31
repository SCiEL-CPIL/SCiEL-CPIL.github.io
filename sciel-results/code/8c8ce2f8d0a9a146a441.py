"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Block Strassen formulas
    # P_s and Q_s are 2x2 matrices of linear combinations of A and B blocks
    # We represent the coefficients of A and B for each element of P_s and Q_s
    # A is 4x4, indexed by (I, a, J, b) where I,J in {0,1} are block indices, a,b in {0,1} are local
    # A[i, j] where i = 2*I + a, j = 2*J + b
    
    # Block Strassen definitions:
    # s=1: P1 = A00 + A11, Q1 = B00 + B11
    # s=2: P2 = A10 + A11, Q2 = B00
    # s=3: P3 = A00,       Q3 = B01 - B11
    # s=4: P4 = A11,       Q4 = B10 - B00
    # s=5: P5 = A00 + A01, Q5 = B11
    # s=6: P6 = A10 - A00, Q6 = B00 + B01
    # s=7: P7 = A11 - A01, Q7 = B10 + B11
    
    # For each s, P_s is a 2x2 matrix. Each element P_s[x,y] is a sum of A blocks.
    # We need to express P_s[x,y] as a linear combination of A's elements.
    
    # Let's define the coefficients for P_s and Q_s.
    # P_s[x, y] = sum_{I,J,a,b} coeff_P[s][x][y][I][J][a][b] * A[2*I+a, 2*J+b]
    # This is too deep. Simpler:
    # P_s is a 2x2 matrix. Its elements are 2x2 matrices (blocks).
    # No, P_s is a 2x2 matrix of SCALARS? No.
    # A00 is a 2x2 matrix. P1 = A00 + A11 is a 2x2 matrix.
    # So P_s is a 2x2 matrix. The elements of P_s are scalars? No, the elements of A00 are scalars.
    # So P_s is a 2x2 matrix of scalars? No, A00 is 2x2. P1 is 2x2.
    # Wait. A is 4x4. A00 is the top-left 2x2.
    # P1 = A00 + A11. This is a 2x2 matrix.
    # So P_s is a 2x2 matrix.
    # Q_s is a 2x2 matrix.
    # R_s = P_s Q_s is a 2x2 matrix.
    
    # Now we apply Scalar Strassen to P_s and Q_s.
    # P_s = [[p00, p01], [p10, p11]]
    # Q_s = [[q00, q01], [q10, q11]]
    
    # Scalar Strassen products M_t:
    # t=1: m1 = p00 * (q00 + q11)
    # t=2: m2 = (p10 + p11) * q00
    # t=3: m3 = p00 * (q01 - q11)
    # t=4: m4 = p11 * (q10 - q00)
    # t=5: m5 = (p00 + p01) * q11
    # t=6: m6 = (p10 - p00) * (q00 + q01)
    # t=7: m7 = (p11 - p01) * (q10 + q11)
    
    # Each M_t is a scalar product of two linear combinations.
    # We need to expand p_xy and q_xy into A and B elements.
    
    # Let's define the expansion of P_s and Q_s elements.
    # P_s[x, y] is a 2x2 matrix? No.
    # In the block Strassen, A00 is a block.
    # So P1 = A00 + A11.
    # A00 is a 2x2 matrix.
    # So P1 is a 2x2 matrix.
    # The elements of P1 are the elements of A00 and A11.
    # P1[0,0] = A00[0,0] + A11[0,0] = A[0,0] + A[2,2]
    # P1[0,1] = A00[0,1] + A11[0,1] = A[0,1] + A[2,3]
    # etc.
    
    # So, for a given s, we can define the 4 elements of P_s (p00, p01, p10, p11) as linear combinations of A's elements.
    # Similarly for Q_s.
    
    # Let's create a helper to get the coefficients.
    # coeff_P[s][x][y] is a list of (index, sign) for A.
    
    def get_P_coeffs(s):
        coeffs = [[[] for _ in range(2)] for _ in range(2)]
        # s is 0-indexed in this helper, so s=0 is s=1 in formula
        if s == 0: # P1 = A00 + A11
            # P1[0,0] = A[0,0] + A[2,2]
            coeffs[0][0] = [(0, 1), (6, 1)] # A[0,0], A[2,2]
            coeffs[0][1] = [(1, 1), (7, 1)] # A[0,1], A[2,3]
            coeffs[1][0] = [(4, 1), (10, 1)] # A[1,0], A[3,2]
            coeffs[1][1] = [(5, 1), (11, 1)] # A[1,1], A[3,3]
        elif s == 1: # P2 = A10 + A11
            coeffs[0][0] = [(4, 1), (10, 1)] # A10[0,0]=A[2,0], A11[0,0]=A[2,2] -> Wait.
            # A10 is bottom-left block: rows 2,3 cols 0,1.
            # A10[0,0] = A[2,0] = index 8? No.
            # Indexing: i=2*I+a, j=2*J+b.
            # A10: I=1, J=0. i=2+a, j=0+b.
            # A10[0,0] -> i=2, j=0 -> idx 8.
            # A11: I=1, J=1. i=2+a, j=2+b.
            # A11[0,0] -> i=2, j=2 -> idx 10.
            coeffs[0][0] = [(8, 1), (10, 1)]
            coeffs[0][1] = [(9, 1), (11, 1)]
            coeffs[1][0] = [(12, 1), (14, 1)]
            coeffs[1][1] = [(13, 1), (15, 1)]
        elif s == 2: # P3 = A00
            coeffs[0][0] = [(0, 1)]
            coeffs[0][1] = [(1, 1)]
            coeffs[1][0] = [(4, 1)]
            coeffs[1][1] = [(5, 1)]
        elif s == 3: # P4 = A11
            coeffs[0][0] = [(10, 1)]
            coeffs[0][1] = [(11, 1)]
            coeffs[1][0] = [(14, 1)]
            coeffs[1][1] = [(15, 1)]
        elif s == 4: # P5 = A00 + A01
            # A00: I=0,J=0. A01: I=0,J=1.
            coeffs[0][0] = [(0, 1), (2, 1)]
            coeffs[0][1] = [(1, 1), (3, 1)]
            coeffs[1][0] = [(4, 1), (6, 1)]
            coeffs[1][1] = [(5, 1), (7, 1)]
        elif s == 5: # P6 = A10 - A00
            coeffs[0][0] = [(8, 1), (0, -1)]
            coeffs[0][1] = [(9, 1), (1, -1)]
            coeffs[1][0] = [(12, 1), (4, -1)]
            coeffs[1][1] = [(13, 1), (5, -1)]
        elif s == 6: # P7 = A11 - A01
            coeffs[0][0] = [(10, 1), (2, -1)]
            coeffs[0][1] = [(11, 1), (3, -1)]
            coeffs[1][0] = [(14, 1), (6, -1)]
            coeffs[1][1] = [(15, 1), (7, -1)]
        return coeffs

    def get_Q_coeffs(s):
        coeffs = [[[] for _ in range(2)] for _ in range(2)]
        if s == 0: # Q1 = B00 + B11
            coeffs[0][0] = [(0, 1), (10, 1)]
            coeffs[0][1] = [(1, 1), (11, 1)]
            coeffs[1][0] = [(4, 1), (14, 1)]
            coeffs[1][1] = [(5, 1), (15, 1)]
        elif s == 1: # Q2 = B00
            coeffs[0][0] = [(0, 1)]
            coeffs[0][1] = [(1, 1)]
            coeffs[1][0] = [(4, 1)]
            coeffs[1][1] = [(5, 1)]
        elif s == 2: # Q3 = B01 - B11
            coeffs[0][0] = [(2, 1), (10, -1)]
            coeffs[0][1] = [(3, 1), (11, -1)]
            coeffs[1][0] = [(6, 1), (14, -1)]
            coeffs[1][1] = [(7, 1), (15, -1)]
        elif s == 3: # Q4 = B10 - B00
            coeffs[0][0] = [(8, 1), (0, -1)]
            coeffs[0][1] = [(9, 1), (1, -1)]
            coeffs[1][0] = [(12, 1), (4, -1)]
            coeffs[1][1] = [(13, 1), (5, -1)]
        elif s == 4: # Q5 = B11
            coeffs[0][0] = [(10, 1)]
            coeffs[0][1] = [(11, 1)]
            coeffs[1][0] = [(14, 1)]
            coeffs[1][1] = [(15, 1)]
        elif s == 5: # Q6 = B00 + B01
            coeffs[0][0] = [(0, 1), (2, 1)]
            coeffs[0][1] = [(1, 1), (3, 1)]
            coeffs[1][0] = [(4, 1), (6, 1)]
            coeffs[1][1] = [(5, 1), (7, 1)]
        elif s == 6: # Q7 = B10 + B11
            coeffs[0][0] = [(8, 1), (10, 1)]
            coeffs[0][1] = [(9, 1), (11, 1)]
            coeffs[1][0] = [(12, 1), (14, 1)]
            coeffs[1][1] = [(13, 1), (15, 1)]
        return coeffs

    # Define Scalar Strassen formulas for P and Q
    # t=0..6
    # We need to form the linear combinations for u and v for each t.
    
    def get_scalar_combos(s, t):
        P_coeffs = get_P_coeffs(s)
        Q_coeffs = get_Q_coeffs(s)
        
        u_combos = []
        v_combos = []
        
        if t == 0: # m1 = p00 * (q00 + q11)
            u_combos = P_coeffs[0][0]
            v_combos = Q_coeffs[0][0] + Q_coeffs[1][1]
        elif t == 1: # m2 = (p10 + p11) * q00
            u_combos = P_coeffs[1][0] + P_coeffs[1][1]
            v_combos = Q_coeffs[0][0]
        elif t == 2: # m3 = p00 * (q01 - q11)
            u_combos = P_coeffs[0][0]
            v_combos = Q_coeffs[0][1] + Q_coeffs[1][1]
            # Apply signs to second part
            for i in range(len(v_combos)):
                if i >= len(Q_coeffs[0][1]):
                    v_combos[i] = (v_combos[i][0], -v_combos[i][1])
        elif t == 3: # m4 = p11 * (q10 - q00)
            u_combos = P_coeffs[1][1]
            v_combos = Q_coeffs[1][0] + Q_coeffs[0][0]
            for i in range(len(v_combos)):
                if i >= len(Q_coeffs[1][0]):
                    v_combos[i] = (v_combos[i][0], -v_combos[i][1])
        elif t == 4: # m5 = (p00 + p01) * q11
            u_combos = P_coeffs[0][0] + P_coeffs[0][1]
            v_combos = Q_coeffs[1][1]
        elif t == 5: # m6 = (p10 - p00) * (q00 + q01)
            u_combos = P_coeffs[1][0] + P_coeffs[0][0]
            for i in range(len(u_combos)):
                if i >= len(P_coeffs[1][0]):
                    u_combos[i] = (u_combos[i][0], -u_combos[i][1])
            v_combos = Q_coeffs[0][0] + Q_coeffs[0][1]
        elif t == 6: # m7 = (p11 - p01) * (q10 + q11)
            u_combos = P_coeffs[1][1] + P_coeffs[0][1]
            for i in range(len(u_combos)):
                if i >= len(P_coeffs[1][1]):
                    u_combos[i] = (u_combos[i][0], -u_combos[i][1])
            v_combos = Q_coeffs[1][0] + Q_coeffs[1][1]
            
        return u_combos, v_combos

    # Define how R_s (2x2) contributes to C (4x4)
    # R_s is a 2x2 matrix.
    # The scalar Strassen result M_t contributes to R_s entries based on standard Strassen output.
    # z00 = m1 + m4 - m5 + m7
    # z01 = m3 + m5
    # z10 = m2 + m4
    # z11 = m1 - m2 + m3 + m6
    
    # Then R_s contributes to C blocks:
    # s=0 (R1): C00 += R1, C11 += R1
    # s=1 (R2): C10 += R2, C11 -= R2
    # s=2 (R3): C01 += R3, C11 += R3
    # s=3 (R4): C00 += R4, C10 += R4
    # s=4 (R5): C00 -= R5, C01 += R5
    # s=5 (R6): C11 += R6
    # s=6 (R7): C00 += R7, C01 -= R7 (Wait, check formula: C00 = R1+R4-R5+R7. C01=R3+R5. C10=R2+R4. C11=R1-R2+R3+R6)
    # Correction on C01: C01 = R3 + R5.
    # Correction on C11: C11 = R1 - R2 + R3 + R6.
    
    # So for each s, we have a mapping from R_s[x,y] to C block.
    # R_s[x,y] is a 2x2 matrix (local indices a,c).
    # C block I,K is a 2x2 matrix.
    # C[2*I+a, 2*K+c] += coeff * R_s[x,y]
    
    # Let's define the block contribution coefficients.
    # block_coeff[s][I][K][x][y] = coeff
    # I,K in {0,1}. x,y in {0,1}.
    
    block_contrib = {}
    for s in range(7):
        block_contrib[s] = {}
        for I in range(2):
            for K in range(2):
                block_contrib[s][I][K] = {}
                for x in range(2):
                    for y in range(2):
                        block_contrib[s][I][K][x][y] = 0

    # s=0 (R1): C00 (I=0,K=0) += R1, C11 (I=1,K=1) += R1
    for x in range(2):
        for y in range(2):
            block_contrib[0][0][0][x][y] = 1
            block_contrib[0][1][1][x][y] = 1
    # s=1 (R2): C10 (I=1,K=0) += R2, C11 (I=1,K=1) -= R2
    for x in range(2):
        for y in range(2):
            block_contrib[1][1][0][x][y] = 1
            block_contrib[1][1][1][x][y] = -1
    # s=2 (R3): C01 (I=0,K=1) += R3, C11 (I=1,K=1) += R3
    for x in range(2):
        for y in range(2):
            block_contrib[2][0][1][x][y] = 1
            block_contrib[2][1][1][x][y] = 1
    # s=3 (R4): C00 (I=0,K=0) += R4, C10 (I=1,K=0) += R4
    for x in range(2):
        for y in range(2):
            block_contrib[3][0][0][x][y] = 1
            block_contrib[3][1][0][x][y] = 1
    # s=4 (R5): C00 (I=0,K=0) -= R5, C01 (I=0,K=1) += R5
    for x in range(2):
        for y in range(2):
            block_contrib[4][0][0][x][y] = -1
            block_contrib[4][0][1][x][y] = 1
    # s=5 (R6): C11 (I=1,K=1) += R6
    for x in range(2):
        for y in range(2):
            block_contrib[5][1][1][x][y] = 1
    # s=6 (R7): C00 (I=0,K=0) += R7, C01 (I=0,K=1) -= R7 (Wait, C01=R3+R5. No R7 in C01?
    # Formula: C00 = R1+R4-R5+R7. C01=R3+R5. C10=R2+R4. C11=R1-R2+R3+R6.
    # So R7 only contributes to C00.
    for x in range(2):
        for y in range(2):
            block_contrib[6][0][0][x][y] = 1

    # Now, for each s, t, we generate the rank-1 vector.
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            u_combos, v_combos = get_scalar_combos(s, t)
            
            # Determine which entries of R_s this M_t affects and with what sign.
            # z00 = m1 + m4 - m5 + m7
            # z01 = m3 + m5
            # z10 = m2 + m4
            # z11 = m1 - m2 + m3 + m6
            
            r_contrib = {}
            if t == 0: # m1
                r_contrib[(0,0)] = 1
                r_contrib[(1,1)] = 1
            elif t == 1: # m2
                r_contrib[(1,0)] = 1
                r_contrib[(1,1)] = -1
            elif t == 2: # m3
                r_contrib[(0,1)] = 1
                r_contrib[(1,1)] = 1
            elif t == 3: # m4
                r_contrib[(0,0)] = 1
                r_contrib[(1,0)] = 1
            elif t == 4: # m5
                r_contrib[(0,0)] = -1
                r_contrib[(0,1)] = 1
            elif t == 5: # m6
                r_contrib[(1,1)] = 1
            elif t == 6: # m7
                r_contrib[(0,0)] = 1

            # For each affected entry (x,y) of R_s, add to U, V, W
            for (x, y), sign_r in r_contrib.items():
                # For each term in u_combos
                # u_combos is a list of (idx, coeff)
                # We need to add to U[idx, r] the value coeff
                # But wait, u_combos is the linear combination for the scalar product.
                # The scalar product is sum(u_i A_i) * sum(v_j B_j).
                # This is not a single rank-1 tensor in A and B.
                # It is a sum of rank-1 tensors?
                # No, the tensor decomposition requires the sum to be a sum of rank-1 tensors.
                # M_t = (sum u_i A_i) (sum v_j B_j) = sum_{i,j} u_i v_j A_i B_j.
                # This is a sum of rank-1 tensors A_i B_j.
                # But we are constructing a decomposition of the *multiplication tensor*.
                # The rank-1 tensors are of the form u \otimes v \otimes w.
                # Here, the "scalar product" M_t is just one scalar multiplication in the Strassen algorithm.
                # In the tensor decomposition, each scalar multiplication corresponds to one rank-1 tensor.
                # So, for M_t, we have one rank-1 tensor.
                # The U vector has non-zeros at the indices of A involved in the first factor.
                # The V vector has non-zeros at the indices of B involved in the second factor.
                # The W vector has non-zeros at the indices of C affected.
                
                # So, for this specific rank-1 tensor (column r):
                # U[idx, r] = coeff_u
                # V[idx, r] = coeff_v
                # W[idx, r] = coeff_w
                
                # But u_combos can have multiple terms.
                # If u_combos has 2 terms, then U has 2 non-zeros.
                # This is fine. A rank-1 tensor u \otimes v \otimes w can have sparse u, v, w.
                
                for idx_u, coeff_u in u_combos:
                    U[idx_u][r] += coeff_u
                for idx_v, coeff_v in v_combos:
                    V[idx_v][r] += coeff_v
                
                # Now for W.
                # The product contributes to R_s[x,y] with sign_r.
                # R_s[x,y] contributes to C blocks.
                # R_s[x,y] is a scalar? No, R_s is a 2x2 matrix.
                # The scalar M_t contributes to the element R_s[x,y].
                # R_s[x,y] is then added to C[2*I+x, 2*K+y] (local indices a,c).
                # Wait, R_s is a 2x2 matrix. Its element (x,y) corresponds to local indices?
                # In Strassen, z00 is the top-left element.
                # So R_s[0,0] corresponds to local (0,0).
                # R_s[0,1] corresponds to local (0,1).
                # etc.
                
                for I in range(2):
                    for K in range(2):
                        coeff_block = block_contrib[s][I][K][x][y]
                        if coeff_block != 0:
                            total_coeff = sign_r * coeff_block
                            # C index: i = 2*I + x, k = 2*K + y
                            # W index: k*n + i
                            i = 2 * I + x
                            k = 2 * K + y
                            w_idx = k * n + i
                            W[w_idx][r] += total_coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
