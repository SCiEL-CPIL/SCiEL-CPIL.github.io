"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-161 decomposition using Strassen's algorithm on 2x2 blocks of 3x3 matrices,
    with Laderman's algorithm (rank 23) for the 3x3 blocks.
    Falls back to rank-189 (Schoolbook) if Laderman verification fails.
    """
    n, m, p = DIMENSIONS
    
    # Define Laderman 3x3 decomposition
    # Returns U3, V3, W3 for 3x3 matrix multiplication
    # U3 shape: (9, 23), V3 shape: (9, 23), W3 shape: (9, 23)
    # Indices: i*3 + j for 3x3 matrices
    U3 = [[0]*23 for _ in range(9)]
    V3 = [[0]*23 for _ in range(9)]
    W3 = [[0]*23 for _ in range(9)]
    
    # Laderman terms (coeffs for A, B, C)
    # Each term: (A_indices, A_coeffs, B_indices, B_coeffs, C_indices, C_coeffs)
    # A_indices are (i, j) in 3x3
    laderman_terms = [
        # 1
        ([(0,0)], [1], [(0,0)], [1], [(0,0)], [1]),
        # 2
        ([(0,1), (0,2)], [1, 1], [(1,0), (2,0)], [1, 1], [(0,1), (0,2)], [1, 1]),
        # 3
        ([(1,0), (2,0)], [1, 1], [(0,1)], [1], [(1,0), (2,0)], [1, 1]),
        # 4
        ([(1,1), (1,2)], [1, 1], [(1,0), (2,0)], [1, 1], [(1,1), (1,2)], [1, 1]),
        # 5
        ([(0,0), (1,0)], [1, 1], [(1,1)], [1], [(0,0), (1,0)], [1, 1]),
        # 6
        ([(0,1), (1,1)], [1, 1], [(0,0), (1,0)], [1, 1], [(0,1), (1,1)], [1, 1]),
        # 7
        ([(0,2), (1,2)], [1, 1], [(0,1), (1,1)], [1, 1], [(0,2), (1,2)], [1, 1]),
        # 8
        ([(2,0), (2,2)], [1, 1], [(1,1)], [1], [(2,0), (2,2)], [1, 1]),
        # 9
        ([(2,1), (2,2)], [1, 1], [(1,0), (2,0)], [1, 1], [(2,1), (2,2)], [1, 1]),
        # 10
        ([(0,0), (2,0)], [1, 1], [(2,1)], [1], [(0,0), (2,0)], [1, 1]),
        # 11
        ([(0,1), (2,1)], [1, 1], [(2,0), (0,0)], [1, 1], [(0,1), (2,1)], [1, 1]),
        # 12
        ([(0,2), (2,2)], [1, 1], [(2,1), (0,1)], [1, 1], [(0,2), (2,2)], [1, 1]),
        # 13
        ([(1,0), (2,0)], [1, 1], [(0,2), (1,2)], [1, 1], [(1,0), (2,0)], [1, 1]),
        # 14
        ([(1,1), (2,1)], [1, 1], [(0,2), (1,2)], [1, 1], [(1,1), (2,1)], [1, 1]),
        # 15
        ([(1,2), (2,2)], [1, 1], [(0,2), (1,2)], [1, 1], [(1,2), (2,2)], [1, 1]),
        # 16
        ([(0,0), (1,1), (2,2)], [1, 1, 1], [(2,2)], [1], [(0,0), (1,1), (2,2)], [1, 1, 1]),
        # 17
        ([(0,0), (1,1)], [1, 1], [(1,2)], [1], [(0,0), (1,1)], [1, 1]),
        # 18
        ([(0,0), (2,2)], [1, 1], [(2,0)], [1], [(0,0), (2,2)], [1, 1]),
        # 19
        ([(0,1), (2,2)], [1, 1], [(2,1)], [1], [(0,1), (2,2)], [1, 1]),
        # 20
        ([(0,2), (1,1)], [1, 1], [(1,2)], [1], [(0,2), (1,1)], [1, 1]),
        # 21
        ([(1,0), (2,2)], [1, 1], [(2,0)], [1], [(1,0), (2,2)], [1, 1]),
        # 22
        ([(1,2), (2,1)], [1, 1], [(2,0)], [1], [(1,2), (2,1)], [1, 1]),
        # 23
        ([(1,2), (2,1)], [1, 1], [(0,2)], [1], [(1,2), (2,1)], [1, 1]),
    ]
    
    # Populate U3, V3, W3
    for k, term in enumerate(laderman_terms):
        a_idx, a_coeff, b_idx, b_coeff, c_idx, c_coeff = term
        for idx, coeff in zip(a_idx, a_coeff):
            i, j = idx
            U3[i*3+j][k] += coeff
        for idx, coeff in zip(b_idx, b_coeff):
            i, j = idx
            V3[i*3+j][k] += coeff
        for idx, coeff in zip(c_idx, c_coeff):
            i, j = idx
            W3[i*3+j][k] += coeff
            
    # Verification of Laderman
    import random
    def verify_laderman():
        for _ in range(5):
            A = [[random.randint(-10, 10) for _ in range(3)] for _ in range(3)]
            B = [[random.randint(-10, 10) for _ in range(3)] for _ in range(3)]
            C_expected = [[sum(A[i][t]*B[t][j] for t in range(3)) for j in range(3)] for i in range(3)]
            C_calc = [[0 for _ in range(3)] for _ in range(3)]
            for k in range(23):
                # Compute M_k
                m_k = 0
                for i in range(3):
                    for j in range(3):
                        m_k += U3[i*3+j][k] * A[i][j] * V3[i*3+j][k] # Wait, U3 and V3 are for linear forms on A and B separately?
                        # No, U3[i*3+j][k] is coeff for A[i][j] in X_k?
                        # My Laderman definition: M_k = X_k * Y_k.
                        # X_k is linear combo of A. Y_k is linear combo of B.
                        # U3 should contain coeffs for X_k. V3 for Y_k.
                        # But I put A coeffs in U3 and B coeffs in V3.
                        # So M_k = (sum A_ij U3_ij_k) * (sum B_ij V3_ij_k).
                        # This is NOT how tensor decomposition works for U, V, W.
                        # In tensor decomposition, M_k = sum U_ik A_i * sum V_jk B_j? No.
                        # M_k = (sum_i U_ik A_i) * (sum_j V_jk B_j).
                        # My U3 and V3 are correct for this.
                x_k = sum(U3[i*3+j][k] * A[i][j] for i in range(3) for j in range(3))
                y_k = sum(V3[i*3+j][k] * B[i][j] for i in range(3) for j in range(3))
                m_k = x_k * y_k
                for i in range(3):
                    for j in range(3):
                        C_calc[i][j] += W3[i*3+j][k] * m_k
            if C_calc != C_expected:
                return False
        return True
        
    use_laderman = verify_laderman()
    
    if not use_laderman:
        # Fallback to Schoolbook 3x3 (Rank 27)
        # We can just return the original 189 rank code
        rank = 189
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        def get_a_index(br, bc, i, j): return (3 * br + i) * 6 + (3 * bc + j)
        def get_b_index(br, bc, i, j): return (3 * br + i) * 6 + (3 * bc + j)
        def get_w_index(r_c, c_c, i, j): return (3 * c_c + j) * 6 + (3 * r_c + i)
        strassen_terms = [
            {'A_blocks': [(0, 0, 1), (1, 1, 1)], 'B_blocks': [(0, 0, 1), (1, 1, 1)], 'C_contrib': [(0, 0, 1), (1, 1, 1)]},
            {'A_blocks': [(1, 0, 1), (1, 1, 1)], 'B_blocks': [(0, 0, 1)], 'C_contrib': [(1, 0, 1), (1, 1, -1)]},
            {'A_blocks': [(0, 0, 1)], 'B_blocks': [(0, 1, 1), (1, 1, -1)], 'C_contrib': [(0, 1, 1), (1, 1, 1)]},
            {'A_blocks': [(1, 1, 1)], 'B_blocks': [(1, 0, 1), (0, 0, -1)], 'C_contrib': [(0, 0, 1), (1, 0, 1)]},
            {'A_blocks': [(0, 0, 1), (0, 1, 1)], 'B_blocks': [(1, 1, 1)], 'C_contrib': [(0, 0, -1), (0, 1, 1)]},
            {'A_blocks': [(1, 0, 1), (0, 0, -1)], 'B_blocks': [(0, 0, 1), (0, 1, 1)], 'C_contrib': [(1, 1, 1)]},
            {'A_blocks': [(0, 1, 1), (1, 1, -1)], 'B_blocks': [(1, 0, 1), (1, 1, 1)], 'C_contrib': [(0, 0, 1)]}
        ]
        for k, term in enumerate(strassen_terms):
            a_blocks = term['A_blocks']
            b_blocks = term['B_blocks']
            c_contrib = term['C_contrib']
            for i in range(3):
                for t in range(3):
                    for j in range(3):
                        for br, bc, coeff in a_blocks:
                            idx = get_a_index(br, bc, i, t)
                            U[idx][col] += coeff
                        for br, bc, coeff in b_blocks:
                            idx = get_b_index(br, bc, t, j)
                            V[idx][col] += coeff
                        for r_c, c_c, coeff in c_contrib:
                            idx = get_w_index(r_c, c_c, i, j)
                            W[idx][col] += coeff
                        col += 1
        return {"U": U, "V": V, "W": W}
        
    # Use Laderman (Rank 23) inside Strassen (Rank 7)
    rank = 7 * 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def get_a_index(br, bc, i, j): return (3 * br + i) * 6 + (3 * bc + j)
    def get_b_index(br, bc, i, j): return (3 * br + i) * 6 + (3 * bc + j)
    def get_w_index(r_c, c_c, i, j): return (3 * c_c + j) * 6 + (3 * r_c + i)
    
    strassen_terms = [
        {'A_blocks': [(0, 0, 1), (1, 1, 1)], 'B_blocks': [(0, 0, 1), (1, 1, 1)], 'C_contrib': [(0, 0, 1), (1, 1, 1)]},
        {'A_blocks': [(1, 0, 1), (1, 1, 1)], 'B_blocks': [(0, 0, 1)], 'C_contrib': [(1, 0, 1), (1, 1, -1)]},
        {'A_blocks': [(0, 0, 1)], 'B_blocks': [(0, 1, 1), (1, 1, -1)], 'C_contrib': [(0, 1, 1), (1, 1, 1)]},
        {'A_blocks': [(1, 1, 1)], 'B_blocks': [(1, 0, 1), (0, 0, -1)], 'C_contrib': [(0, 0, 1), (1, 0, 1)]},
        {'A_blocks': [(0, 0, 1), (0, 1, 1)], 'B_blocks': [(1, 1, 1)], 'C_contrib': [(0, 0, -1), (0, 1, 1)]},
        {'A_blocks': [(1, 0, 1), (0, 0, -1)], 'B_blocks': [(0, 0, 1), (0, 1, 1)], 'C_contrib': [(1, 1, 1)]},
        {'A_blocks': [(0, 1, 1), (1, 1, -1)], 'B_blocks': [(1, 0, 1), (1, 1, 1)], 'C_contrib': [(0, 0, 1)]}
    ]
    
    col = 0
    for s_k, term in enumerate(strassen_terms):
        a_blocks = term['A_blocks']
        b_blocks = term['B_blocks']
        c_contrib = term['C_contrib']
        
        # For each of the 23 Laderman terms
        for l_k in range(23):
            # U: Linear form for X_lk from A blocks
            # X_lk is sum of A_ij * U3_ij_lk
            # But A here is block A_br_bc
            # X_lk = sum_{br,bc} sum_{i,j} U3_ij_lk * (sum_{br,bc} coeff_a * A_br_bc[i,j])
            # X_lk = sum_{br,bc,i,j} (U3_ij_lk * coeff_a) * A_br_bc[i,j]
            
            # We need to fill U[global_a_idx][col]
            # global_a_idx corresponds to A_br_bc[i,j]
            
            # For each i, j in 3x3:
            for i in range(3):
                for j in range(3):
                    coeff_u3 = U3[i*3+j][l_k]
                    if coeff_u3 == 0: continue
                    for br, bc, coeff_a in a_blocks:
                        idx = get_a_index(br, bc, i, j)
                        U[idx][col] += coeff_u3 * coeff_a
                        
            # V: Linear form for Y_lk from B blocks
            for i in range(3):
                for j in range(3):
                    coeff_v3 = V3[i*3+j][l_k]
                    if coeff_v3 == 0: continue
                    for br, bc, coeff_b in b_blocks:
                        idx = get_b_index(br, bc, i, j)
                        V[idx][col] += coeff_v3 * coeff_b
                        
            # W: Contribution to C
            # M_lk contributes to C_block[r_c, c_c] with coeff_c
            # C_block[r_c, c_c] is 3x3 matrix Z_lk
            # Z_lk = sum_{i,j} W3_ij_lk * M_lk
            # So C_global[r_c, c_c, i, j] += coeff_c * W3_ij_lk * M_lk
            
            for r_c, c_c, coeff_c in c_contrib:
                for i in range(3):
                    for j in range(3):
                        coeff_w3 = W3[i*3+j][l_k]
                        if coeff_w3 == 0: continue
                        idx = get_w_index(r_c, c_c, i, j)
                        W[idx][col] += coeff_w3 * coeff_c
                        
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
