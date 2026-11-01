"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will use a rank 36 decomposition.
    # Split A (2x4) into A_L (2x2) and A_R (2x2).
    # Split B (4x5) into B_L (2x5) and B_R (2x5).
    # C = A_L B_L + A_R B_R.
    # Rank of 2x2x5 is 18. Total 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)] # 8 x 36
    V = [[0] * rank for _ in range(m * p)] # 20 x 36
    W = [[0] * rank for _ in range(p * n)] # 10 x 36
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        # u_idx: index in U (0..7)
        # v_idx: index in V (0..19)
        # w_idx: index in W (0..9)
        # coeff: value
        pass

    # We need to generate the 18 terms for a 2x2x5 multiplication
    # and apply them to (A_L, B_L) and (A_R, B_R).
    
    # Indices for 2x2 A block:
    # A_L: rows 0,1; cols 0,1.
    # U indices for A_L: 0 (0,0), 1 (0,1), 2 (1,0), 3 (1,1)
    # A_R: rows 0,1; cols 2,3.
    # U indices for A_R: 4 (0,2), 5 (0,3), 6 (1,2), 7 (1,3)
    
    # Indices for 2x5 B block:
    # B_L: rows 0,1; cols 0..4.
    # V indices for B_L:
    # Row 0: 0 (0,0), 1 (0,1), 2 (0,2), 3 (0,3), 4 (0,4)
    # Row 1: 5 (1,0), 6 (1,1), 7 (1,2), 8 (1,3), 9 (1,4)
    # B_R: rows 2,3; cols 0..4.
    # V indices for B_R:
    # Row 0 (global row 2): 10 (2,0), 11 (2,1), 12 (2,2), 13 (2,3), 14 (2,4)
    # Row 1 (global row 3): 15 (3,0), 16 (3,1), 17 (3,2), 18 (3,3), 19 (3,4)
    
    # W indices for 2x5 C block:
    # C_L: rows 0,1; cols 0..4.
    # W[k*n+i] -> W[5*k + i]
    # k=0: 0 (0,0), 1 (0,1)
    # k=1: 2 (1,0), 3 (1,1)
    # k=2: 4 (2,0), 5 (2,1)
    # k=3: 6 (3,0), 7 (3,1)
    # k=4: 8 (4,0), 9 (4,1)
    # C_R: same W indices?
    # No, C is the sum. So W indices are the same.
    # C = C_L + C_R.
    # So we add to the same W columns.
    
    # Helper to add a Strassen 2x2x2 block contribution
    # local_u: list of 4 indices into U for the 2x2 block
    # local_v: list of 5 indices into V for the 2x5 block? 
    # Actually, Strassen is for 2x2x2.
    # We decompose 2x2x5 into 2x2x2 (cols 0,1) + 2x2x2 (cols 2,3) + 2x2x1 (col 4).
    
    def add_strassen_222(u_idx_map, v_idx_map, w_idx_map, offset):
        # u_idx_map: dict {0: u00, 1: u01, 2: u10, 3: u11}
        # v_idx_map: dict {0: v00, 1: v01, 2: v10, 3: v11}
        # w_idx_map: dict {0: w00, 1: w01, 2: w10, 3: w11}
        
        terms = [
            # (P1, (u11+u22), (v11+v22)) -> W11, W22? No, P1 contributes to multiple Ws.
            # Let's use the standard Strassen coefficients.
            # P1 = (A11+A22)(B11+B22)
            # P2 = (A21+A22)B11
            # P3 = A11(B12-B22)
            # P4 = A22(B21-B11)
            # P5 = (A11+A12)B22
            # P6 = (A21-A11)(B11+B12)
            # P7 = (A12-A22)(B21+B22)
            
            # M11 = P1 + P4 - P5 + P7
            # M12 = P3 + P5
            # M21 = P2 + P4
            # M22 = P1 - P2 + P3 + P6
            
            # We need to emit 7 terms.
            # Each term is a product of a linear combo of U and a linear combo of V.
            # The result is added to W.
            
            # Term 1: P1
            # U part: A11 + A22 -> u_idx_map[0] + u_idx_map[3]
            # V part: B11 + B22 -> v_idx_map[0] + v_idx_map[3]
            # W contributions:
            # W11 (M11) += 1
            # W22 (M22) += 1
            
            # This is getting complex to manage with a helper that just takes indices.
            # It's easier to just list the 7 (U, V, W) tuples with coefficients.
        ]
        pass

    # Let's manually define the 18 terms for one 2x2x5 block.
    # Then apply to A_L/B_L and A_R/B_R.
    
    # A_L u-indices: 0,1,2,3
    # B_L v-indices: 0..4 (cols 0..4).
    # Wait, Strassen splits B into 2x2, 2x2, 2x1.
    # B_L cols 0,1 -> V 0,1 (row 0), 5,6 (row 1).
    # B_L cols 2,3 -> V 2,3 (row 0), 7,8 (row 1).
    # B_L col 4 -> V 4 (row 0), 9 (row 1).
    
    # Let's define the terms for the first 2x2x2 block (cols 0,1 of B).
    # U: 0,1,2,3
    # V: 0,1,5,6 (B11=0, B12=1, B21=5, B22=6)
    # W: 0,1,2,3 (M11=0, M12=1, M21=2, M22=3)
    
    # P1: U(0)+U(3), V(0)+V(5) -> W0(1), W3(1)
    # P2: U(2)+U(3), V(0) -> W2(1)
    # P3: U(0), V(1)-V(6) -> W1(1)
    # P4: U(3), V(5)-V(0) -> W0(1), W2(1)
    # P5: U(0)+U(1), V(6) -> W1(1), W0(-1)
    # P6: U(2)-U(0), V(0)+V(1) -> W3(1)
    # P7: U(1)-U(3), V(5)+V(6) -> W0(1), W3(1)
    
    # This is error prone. I will write a generic generator for Strassen 2x2x2.
    
    def generate_strassen_222(u_base, v_base, w_base, term_offset):
        # u_base: list of 4 indices [u00, u01, u10, u11]
        # v_base: list of 4 indices [v00, v01, v10, v11]
        # w_base: list of 4 indices [w00, w01, w10, w11]
        
        # Returns list of (u_indices_with_coeffs, v_indices_with_coeffs, w_indices_with_coeffs)
        # Actually, we need to add to the global matrices.
        # We will return a list of tuples: 
        # (list of (u_idx, coeff), list of (v_idx, coeff), list of (w_idx, coeff))
        
        terms = []
        
        # P1
        u1 = [(u_base[0], 1), (u_base[3], 1)]
        v1 = [(v_base[0], 1), (v_base[3], 1)]
        w1 = [(w_base[0], 1), (w_base[3], 1)]
        terms.append((u1, v1, w1))
        
        # P2
        u2 = [(u_base[2], 1), (u_base[3], 1)]
        v2 = [(v_base[0], 1)]
        w2 = [(w_base[2], 1)]
        terms.append((u2, v2, w2))
        
        # P3
        u3 = [(u_base[0], 1)]
        v3 = [(v_base[1], 1), (v_base[3], -1)]
        w3 = [(w_base[1], 1)]
        terms.append((u3, v3, w3))
        
        # P4
        u4 = [(u_base[3], 1)]
        v4 = [(v_base[2], 1), (v_base[0], -1)]
        w4 = [(w_base[0], 1), (w_base[2], 1)]
        terms.append((u4, v4, w4))
        
        # P5
        u5 = [(u_base[0], 1), (u_base[1], 1)]
        v5 = [(v_base[3], 1)]
        w5 = [(w_base[1], 1), (w_base[0], -1)]
        terms.append((u5, v5, w5))
        
        # P6
        u6 = [(u_base[2], 1), (u_base[0], -1)]
        v6 = [(v_base[0], 1), (v_base[1], 1)]
        w6 = [(w_base[3], 1)]
        terms.append((u6, v6, w6))
        
        # P7
        u7 = [(u_base[1], 1), (u_base[3], -1)]
        v7 = [(v_base[2], 1), (v_base[3], 1)]
        w7 = [(w_base[0], 1), (w_base[3], 1)]
        terms.append((u7, v7, w7))
        
        return terms

    def add_2x2x1(u_base, v_base, w_base):
        # C11 = A11 V00 + A12 V10
        # C12 = A11 V01 + A12 V11
        # C21 = A21 V00 + A22 V10
        # C22 = A21 V01 + A22 V11
        # Wait, 2x2x1 means B is 2x1.
        # A is 2x2. B is 2x1. C is 2x1.
        # C1 = A11 B1 + A12 B2
        # C2 = A21 B1 + A22 B2
        # This is rank 4.
        # v_base: [v0, v1]
        # w_base: [w0, w1]
        
        terms = []
        terms.append([[(u_base[0], 1)], [(v_base[0], 1)], [(w_base[0], 1)]])
        terms.append([[(u_base[1], 1)], [(v_base[1], 1)], [(w_base[0], 1)]])
        terms.append([[(u_base[2], 1)], [(v_base[0], 1)], [(w_base[1], 1)]])
        terms.append([[(u_base[3], 1)], [(v_base[1], 1)], [(w_base[1], 1)]])
        return terms

    # Apply to A_L, B_L
    # A_L u: 0,1,2,3
    # B_L v: 
    #   Block 1 (cols 0,1): v00=0, v01=1, v10=5, v11=6
    #   Block 2 (cols 2,3): v00=2, v01=3, v10=7, v11=8
    #   Block 3 (col 4): v0=4, v1=9
    # W (same for L and R):
    #   Block 1 (cols 0,1): w00=0, w01=1, w10=2, w11=3
    #   Block 2 (cols 2,3): w00=4, w01=5, w10=6, w11=7
    #   Block 3 (col 4): w0=8, w1=9

    # Generate terms for L
    terms_L = []
    terms_L += generate_strassen_222([0,1,2,3], [0,1,5,6], [0,1,2,3], 0)
    terms_L += generate_strassen_222([0,1,2,3], [2,3,7,8], [4,5,6,7], 0)
    terms_L += add_2x2x1([0,1,2,3], [4,9], [8,9])
    
    # Generate terms for R
    # A_R u: 4,5,6,7
    # B_R v:
    #   Block 1 (cols 0,1): v00=10, v01=11, v10=15, v11=16
    #   Block 2 (cols 2,3): v00=12, v01=13, v10=17, v11=18
    #   Block 3 (col 4): v0=14, v1=19
    # W same.
    
    terms_R = []
    terms_R += generate_strassen_222([4,5,6,7], [10,11,15,16], [0,1,2,3], 0)
    terms_R += generate_strassen_222([4,5,6,7], [12,13,17,18], [4,5,6,7], 0)
    terms_R += add_2x2x1([4,5,6,7], [14,19], [8,9])
    
    all_terms = terms_L + terms_R
    
    # Fill U, V, W
    for idx, (u_part, v_part, w_part) in enumerate(all_terms):
        for u_idx, coeff in u_part:
            U[u_idx][idx] += coeff
        for v_idx, coeff in v_part:
            V[v_idx][idx] += coeff
        for w_idx, coeff in w_part:
            W[w_idx][idx] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
