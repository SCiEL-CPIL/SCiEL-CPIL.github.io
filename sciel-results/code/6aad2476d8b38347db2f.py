import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Rank 49 decomposition for 4x4 matrix multiplication
    # We will generate the 49 terms.
    
    def get_4x4_terms():
        terms = []
        
        # Strassen block definitions
        # S1: X = A00+A11, Y = B00+B11
        # S2: X = A10+A11, Y = B00
        # S3: X = A00, Y = B01-B11
        # S4: X = A11, Y = B10-B00
        # S5: X = A00+A01, Y = B11
        # S6: X = A10-A00, Y = B00+B01
        # S7: X = A01-A11, Y = B10+B11
        
        # C blocks:
        # C00 = S1 + S4 - S5 + S7
        # C01 = S3 + S5
        # C10 = S2 + S4
        # C11 = S1 - S2 + S3 + S6
        
        # For each S_i, we have 7 scalar products P_{i,1}..P_{i,7}.
        # Z_i = X_i Y_i.
        # Z_i[0][0] = P1 + P4 - P5 + P7
        # Z_i[0][1] = P3 + P5
        # Z_i[1][0] = P2 + P4
        # Z_i[1][1] = P1 - P2 + P3 + P6
        
        # We need to map P_{i,j} to C entries.
        
        # Define the 7 products for a generic X, Y.
        # P1 = (X00+X11)(Y00+Y11)
        # P2 = (X10+X11)Y00
        # P3 = X00(Y01-Y11)
        # P4 = X11(Y10-Y00)
        # P5 = (X00+X01)Y11
        # P6 = (X10-X00)(Y00+Y01)
        # P7 = (X01-X11)(Y10+Y11)
        
        # We will create a list of 49 tuples (U, V, W).
        # U, V, W are vectors of length 16.
        
        # Helper to add a term
        def add_term(u, v, w):
            terms.append((u, v, w))
            
        # Define block indices
        # A blocks: (br, bc)
        # A entry (i, j) where i = br*2 + lr, j = bc*2 + lc
        
        # S_defs: (X_blocks, Y_blocks)
        S_defs = [
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]), # S1
            ([(1,0,1), (1,1,1)], [(0,0,1)]), # S2
            ([(0,0,1)], [(0,1,1), (1,1,-1)]), # S3
            ([(1,1,1)], [(1,0,1), (0,0,-1)]), # S4
            ([(0,0,1), (0,1,1)], [(1,1,1)]), # S5
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]), # S6
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]) # S7
        ]
        
        # C_block_coeffs: dict C_block -> list of (S_idx, coeff)
        # C00: S1(1), S4(1), S5(-1), S7(1)
        # C01: S3(1), S5(1)
        # C10: S2(1), S4(1)
        # C11: S1(1), S2(-1), S3(1), S6(1)
        
        C_block_coeffs = {
            (0,0): [(0,1), (3,1), (4,-1), (6,1)],
            (0,1): [(2,1), (4,1)],
            (1,0): [(1,1), (3,1)],
            (1,1): [(0,1), (1,-1), (2,1), (5,1)]
        }
        
        # Z_entry_coeffs: dict Z_entry -> list of (P_idx, coeff)
        # Z00: P1(1), P4(1), P5(-1), P7(1)
        # Z01: P3(1), P5(1)
        # Z10: P2(1), P4(1)
        # Z11: P1(1), P2(-1), P3(1), P6(1)
        
        Z_entry_coeffs = {
            (0,0): [(0,1), (3,1), (4,-1), (6,1)],
            (0,1): [(2,1), (4,1)],
            (1,0): [(1,1), (3,1)],
            (1,1): [(0,1), (1,-1), (2,1), (5,1)]
        }
        
        # For each S_i
        for s_idx, (X_blocks, Y_blocks) in enumerate(S_defs):
            # Build X and Y coefficient vectors for the 4 entries
            # X[0][0], X[0][1], X[1][0], X[1][1]
            # Each is a vector of 16 coeffs for A.
            
            X_coeffs = [[0]*16 for _ in range(4)] # 4 entries, 16 coeffs
            Y_coeffs = [[0]*16 for _ in range(4)]
            
            for br, bc, coeff in X_blocks:
                for lr in range(2):
                    for lc in range(2):
                        i = br*2 + lr
                        j = bc*2 + lc
                        X_coeffs[lr*2+lc][i*4+j] = coeff # Wait, X is 2x2.
                        # X[lr][lc] is the entry.
                        # X[lr][lc] gets coeff * A[i][j]
                        
            for br, bc, coeff in Y_blocks:
                for lr in range(2):
                    for lc in range(2):
                        i = br*2 + lr
                        j = bc*2 + lc
                        Y_coeffs[lr*2+lc][i*4+j] = coeff
                        
            # Now generate 7 products
            # P1 = (X00+X11)(Y00+Y11)
            # u = X00 + X11
            # v = Y00 + Y11
            
            # We need to sum the coefficient vectors.
            
            def sum_vecs(vecs):
                res = [0]*16
                for v in vecs:
                    for k in range(16):
                        res[k] += v[k]
                return res
                
            # P1
            u1 = sum_vecs([X_coeffs[0], X_coeffs[3]])
            v1 = sum_vecs([Y_coeffs[0], Y_coeffs[3]])
            
            # P2
            u2 = sum_vecs([X_coeffs[2], X_coeffs[3]])
            v2 = Y_coeffs[0]
            
            # P3
            u3 = X_coeffs[0]
            v3 = sum_vecs([Y_coeffs[1], [-x for x in Y_coeffs[3]]])
            
            # P4
            u4 = X_coeffs[3]
            v4 = sum_vecs([Y_coeffs[2], [-x for x in Y_coeffs[0]]])
            
            # P5
            u5 = sum_vecs([X_coeffs[0], X_coeffs[1]])
            v5 = Y_coeffs[3]
            
            # P6
            u6 = sum_vecs([X_coeffs[2], [-x for x in X_coeffs[0]]])
            v6 = sum_vecs([Y_coeffs[0], Y_coeffs[1]])
            
            # P7
            u7 = sum_vecs([X_coeffs[1], [-x for x in X_coeffs[3]]])
            v7 = sum_vecs([Y_coeffs[2], Y_coeffs[3]])
            
            P_u = [u1, u2, u3, u4, u5, u6, u7]
            P_v = [v1, v2, v3, v4, v5, v6, v7]
            
            # Now map to C
            # For each Z entry (lr, lc)
            for lr in range(2):
                for lc in range(2):
                    # Z[lr][lc] = sum P_k * coeff
                    # Z[lr][lc] is an entry of S_i.
                    # S_i contributes to C blocks.
                    
                    # Get coeffs for Z[lr][lc]
                    z_coeffs = Z_entry_coeffs[(lr, lc)]
                    
                    # For each P_k in z_coeffs
                    for p_idx, coeff in z_coeffs:
                        # P_{p_idx} contributes to Z[lr][lc] with coeff
                        # Z[lr][lc] is at position (br, bc) in C?
                        # No, S_i is a 2x2 block.
                        # S_i is not a block of C. S_i is one of the 7 Strassen products.
                        # C blocks are linear combinations of S_i.
                        
                        # So we need to sum over C blocks that contain S_i.
                        
                        # For each C block (cb_r, cb_c)
                        for cb_r in range(2):
                            for cb_c in range(2):
                                if (cb_r, cb_c) in C_block_coeffs:
                                    # Check if S_i is in C_block_coeffs[(cb_r, cb_c)]
                                    s_coeff = 0
                                    for s_idx_in_c, c_val in C_block_coeffs[(cb_r, cb_c)]:
                                        if s_idx_in_c == s_idx:
                                            s_coeff = c_val
                                            
                                    if s_coeff != 0:
                                        # Total coeff for P_{p_idx} in C[cb_r*2+lr, cb_c*2+lc]
                                        total_coeff = coeff * s_coeff
                                        
                                        # Create W vector
                                        w = [0]*16
                                        i = cb_r*2 + lr
                                        k = cb_c*2 + lc
                                        w[k*4+i] = total_coeff
                                        
                                        add_term(P_u[p_idx], P_v[p_idx], w)
                                        
        return terms

    terms_4x4 = get_4x4_terms()
    
    # Now build the full decomposition for 4x9
    # B1: cols 0-3 (4x4)
    # B2: cols 4-7 (4x4)
    # B3: col 8 (4x1)
    
    all_terms = []
    
    # Add 49 terms for B1
    for u, v, w in terms_4x4:
        # v is 16x1 for B1 (cols 0-3)
        # We need to map to V of size 36.
        # V index j*9+k.
        # v index j*4+k.
        # k in 0-3.
        v_full = [0]*36
        for j in range(4):
            for k in range(4):
                v_full[j*9+k] = v[j*4+k]
        all_terms.append((u, v_full, w))
        
    # Add 49 terms for B2
    for u, v, w in terms_4x4:
        # v is 16x1 for B2 (cols 4-7)
        v_full = [0]*36
        for j in range(4):
            for k in range(4):
                v_full[j*9+(k+4)] = v[j*4+k]
        # w is for C cols 4-7
        w_full = [0]*36
        for k in range(4):
            for i in range(4):
                w_full[(k+4)*4+i] = w[k*4+i]
        all_terms.append((u, v_full, w_full))
        
    # Add 16 terms for B3 (schoolbook)
    for i in range(4):
        for j in range(4):
            u = [0]*16
            u[i*4+j] = 1
            v = [0]*36
            v[j*9+8] = 1
            w = [0]*36
            w[8*4+i] = 1
            all_terms.append((u, v, w))
            
    rank = len(all_terms)
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    for r, (u, v, w) in enumerate(all_terms):
        for i in range(16):
            U[i][r] = u[i]
        for j in range(36):
            V[j][r] = v[j]
        for k in range(36):
            W[k][r] = w[k]
            
    return {"U": U, "V": V, "W": W}
