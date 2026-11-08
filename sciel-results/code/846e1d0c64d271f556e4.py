"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrices
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # Products:
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b01-b11)
    # P4 = a11(b10-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b01)
    # P7 = (a01-a11)(b10+b11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    # Top-level Strassen for 4x4
    # A blocks: A11 (0,0-1,1), A12 (0,2-1,3), A21 (2,0-3,1), A22 (2,2-3,3)
    # B blocks: B11, B12, B21, B22
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map the 2x2 Strassen decomposition to the 4x4 blocks.
    # Each Mk is a 2x2 matrix product X*Y.
    # X and Y are 2x2 matrices composed of linear combinations of A/B blocks.
    
    def get_block_indices(block):
        # Returns list of (row, col) for the 2x2 block
        # A11: (0,0), (0,1), (1,0), (1,1)
        # A12: (0,2), (0,3), (1,2), (1,3)
        # A21: (2,0), (2,1), (3,0), (3,1)
        # A22: (2,2), (2,3), (3,2), (3,3)
        if block == 11: return [(0,0), (0,1), (1,0), (1,1)]
        if block == 12: return [(0,2), (0,3), (1,2), (1,3)]
        if block == 21: return [(2,0), (2,1), (3,0), (3,1)]
        if block == 22: return [(2,2), (2,3), (3,2), (3,3)]
        return []

    # Define the 7 top-level terms M1..M7
    # Each term is (X_block_ops, Y_block_ops)
    # X_block_ops: list of (coeff, block_id)
    # Y_block_ops: list of (coeff, block_id)
    
    M_terms = [
        # M1
        ([(1, 11), (1, 22)], [(1, 11), (1, 22)]),
        # M2
        ([(1, 21), (1, 22)], [(1, 11)]),
        # M3
        ([(1, 11)], [(1, 12), (-1, 22)]),
        # M4
        ([(1, 22)], [(1, 21), (-1, 11)]),
        # M5
        ([(1, 11), (1, 12)], [(1, 22)]),
        # M6
        ([(1, 21), (-1, 11)], [(1, 11), (1, 12)]),
        # M7
        ([(1, 12), (-1, 22)], [(1, 21), (1, 22)])
    ]
    
    # C block contributions
    # C11 gets M1, M4, -M5, M7
    # C12 gets M3, M5
    # C21 gets M2, M4
    # C22 gets M1, -M2, M3, M6
    
    C_contribs = [
        [(1, 11), (1, 22), (-1, 21), (1, 12)], # C11: M1, M4, -M5, M7 ? No, indices 0..6
        # Let's use indices 0..6 for M1..M7
        # C11: M1(0), M4(3), -M5(4), M7(6)
        [(1, 0), (1, 3), (-1, 4), (1, 6)],
        # C12: M3(2), M5(4)
        [(1, 2), (1, 4)],
        # C21: M2(1), M4(3)
        [(1, 1), (1, 3)],
        # C22: M1(0), -M2(1), M3(2), M6(5)
        [(1, 0), (-1, 1), (1, 2), (1, 5)]
    ]
    
    # Strassen 2x2 coefficients
    # P1: A: (1,00), (1,11); B: (1,00), (1,11)
    # P2: A: (1,10), (1,11); B: (1,00)
    # P3: A: (1,00); B: (1,01), (-1,11)
    # P4: A: (1,11); B: (1,10), (-1,00)
    # P5: A: (1,00), (1,01); B: (1,11)
    # P6: A: (1,10), (-1,00); B: (1,00), (1,01)
    # P7: A: (1,01), (-1,11); B: (1,10), (1,11)
    
    S_A = [
        [(1, 0), (1, 3)], # P1
        [(1, 2), (1, 3)], # P2
        [(1, 0)],         # P3
        [(1, 3)],         # P4
        [(1, 0), (1, 1)], # P5
        [(1, 2), (-1, 0)],# P6
        [(1, 1), (-1, 3)] # P7
    ]
    S_B = [
        [(1, 0), (1, 3)], # P1
        [(1, 0)],         # P2
        [(1, 1), (-1, 3)],# P3
        [(1, 2), (-1, 0)],# P4
        [(1, 3)],         # P5
        [(1, 0), (1, 1)], # P6
        [(1, 2), (1, 3)]  # P7
    ]
    
    # C 2x2 contributions from P1..P7
    # C00: P1, P4, -P5, P7
    # C01: P3, P5
    # C10: P2, P4
    # C11: P1, -P2, P3, P6
    
    S_C = [
        [(1, 0), (1, 3), (-1, 4), (1, 6)], # C00
        [(1, 2), (1, 4)],                  # C01
        [(1, 1), (1, 3)],                  # C10
        [(1, 0), (-1, 1), (1, 2), (1, 5)]  # C11
    ]
    
    # Map 2x2 block indices to 4x4 scalar indices
    # Block 11: 0,1,4,5 (row-major: 0,0->0; 0,1->1; 1,0->4; 1,1->5)
    # Block 12: 2,3,6,7
    # Block 21: 8,9,12,13
    # Block 22: 10,11,14,15
    
    def block_to_scalar_indices(block_id, local_idx):
        # local_idx 0..3 corresponds to 00, 01, 10, 11
        if block_id == 11:
            return [0, 1, 4, 5][local_idx]
        if block_id == 12:
            return [2, 3, 6, 7][local_idx]
        if block_id == 21:
            return [8, 9, 12, 13][local_idx]
        if block_id == 22:
            return [10, 11, 14, 15][local_idx]
        return -1

    # Construct U, V, W
    rank_idx = 0
    for m_idx in range(7):
        X_ops, Y_ops = M_terms[m_idx]
        for s_idx in range(7):
            # This is term (m_idx, s_idx)
            # X is formed by X_ops, Y by Y_ops
            # The product is X*Y, decomposed by S_A[s_idx], S_B[s_idx]
            
            # U part: Linear combination of A entries
            # V part: Linear combination of B entries
            # W part: Contribution to C entries
            
            # We need to add the contribution of this specific multiplication
            # to the U, V, W matrices at column rank_idx.
            
            # U: Sum of coefficients for A
            # V: Sum of coefficients for B
            # W: Sum of coefficients for C
            
            # The multiplication is (Sum_A) * (Sum_B)
            # Sum_A = sum_{(c, b) in X_ops} sum_{(c_s, l) in S_A[s_idx]} c * c_s * A_{block(b, l)}
            # Sum_B = sum_{(c, b) in Y_ops} sum_{(c_s, l) in S_B[s_idx]} c * c_s * B_{block(b, l)}
            
            # Wait, S_A and S_B are for the 2x2 product of X and Y.
            # X is a 2x2 matrix. X[0,0] is a scalar (linear comb of A).
            # S_A[s_idx] gives coefficients for X[0,0], X[0,1], etc.
            
            # Let's expand U
            U_vec = [0.0] * 16
            V_vec = [0.0] * 16
            W_vec = [0.0] * 16
            
            # Calculate U_vec
            for c_x, b_x in X_ops:
                for c_s, l_s in S_A[s_idx]:
                    # l_s is 0..3 (00, 01, 10, 11)
                    idx = block_to_scalar_indices(b_x, l_s)
                    U_vec[idx] += c_x * c_s
            
            # Calculate V_vec
            for c_y, b_y in Y_ops:
                for c_s, l_s in S_B[s_idx]:
                    idx = block_to_scalar_indices(b_y, l_s)
                    V_vec[idx] += c_y * c_s
            
            # Calculate W_vec
            # The result of the 2x2 product X*Y is a 2x2 matrix R.
            # R is added to C blocks according to C_contribs.
            # First, find which C block R contributes to.
            # m_idx is the index of M (0..6).
            # C_contribs[c_block_idx] contains (coeff, m_idx).
            # We need to find which c_block_idx has m_idx in its list.
            
            # Actually, it's easier to iterate C blocks.
            # For each C block (11, 12, 21, 22), check if it depends on M[m_idx].
            # If so, get the coefficient coeff_c.
            # Then, R (the 2x2 result of X*Y) contributes to that C block.
            # The entries of R are linear combinations of P1..P7.
            # S_C gives the combination.
            
            # R[0,0] = sum_{(c_p, p_idx) in S_C[0]} c_p * P[p_idx]
            # Here P[p_idx] is the scalar product we are currently processing?
            # No. P[p_idx] is the result of the p_idx-th Strassen term for the 2x2 product X*Y.
            # We are currently processing term s_idx of the 2x2 product.
            # So only if p_idx == s_idx does it contribute.
            
            # So for each C block:
            c_block_ids = [11, 12, 21, 22]
            for c_b_idx, c_b_id in enumerate(c_block_ids):
                # Check if C[c_b_id] depends on M[m_idx]
                coeff_m = 0
                for c_m, m_dep in C_contribs[c_b_idx]:
                    if m_dep == m_idx:
                        coeff_m += c_m
                
                if coeff_m != 0:
                    # C[c_b_id] gets coeff_m * R
                    # R is the 2x2 matrix resulting from X*Y.
                    # We need to add coeff_m * R to C[c_b_id].
                    # R's entries are determined by S_C.
                    # R[l_r] (l_r 0..3) = sum_{(c_p, p_idx) in S_C[l_r]} c_p * P[p_idx]
                    # We are at P[s_idx]. So only p_idx == s_idx contributes.
                    
                    for l_r in range(4):
                        # Find coefficient of P[s_idx] in R[l_r]
                        coeff_r = 0
                        for c_p, p_idx in S_C[l_r]:
                            if p_idx == s_idx:
                                coeff_r += c_p
                        
                        if coeff_r != 0:
                            # Total coefficient for this scalar multiplication in C[c_b_id, l_r]
                            total_coeff = coeff_m * coeff_r
                            idx = block_to_scalar_indices(c_b_id, l_r)
                            W_vec[idx] += total_coeff
            
            # Assign to U, V, W
            for i in range(16):
                U[i][rank_idx] = U_vec[i]
                V[i][rank_idx] = V_vec[i]
                W[i][rank_idx] = W_vec[i]
            
            rank_idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
