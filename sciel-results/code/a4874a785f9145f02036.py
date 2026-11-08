import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Block definitions
    # Block (bi, bj) contains entries with global indices:
    # (bi*2 + li, bj*2 + lj) for li, lj in {0, 1}
    def block_indices(bi, bj):
        return [ (bi*2 + li)*4 + (bj*2 + lj) for li in range(2) for lj in range(2) ]
    
    # Strassen 2x2 formulas
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # S1 = (x00+x11)(y00+y11)
    # S2 = (x10+x11)y00
    # S3 = x00(y01-y11)
    # S4 = x11(y10-y00)
    # S5 = (x00+x01)y11
    # S6 = (x10-x00)(y00+y01)
    # S7 = (x01-x11)(y10+y11)
    
    # Reconstruction of Z = XY
    # z00 = S1 + S4 - S5 + S7
    # z01 = S3 + S5
    # z10 = S2 + S4
    # z11 = S1 - S2 + S3 + S6
    
    # Outer Strassen for 4x4
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Reconstruction of C
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the 7 outer products
    # Each product is defined by X_blocks and Y_blocks
    # X_blocks is a list of (coeff, block_idx) pairs for the 4 entries of the 2x2 X matrix?
    # No, X is a 2x2 matrix of blocks.
    # Let's represent X as a 2x2 array of block coefficients.
    # X[i][j] is a vector of coefficients for the blocks A_{kl} involved in X_{ij}.
    # Actually, each entry of X is a linear combination of blocks.
    
    # Let's define a helper to create the coefficient vectors for U and V for a specific term.
    
    # We will iterate through the 49 terms.
    for c in range(rank):
        k = c // 7
        l = c % 7
        
        # Determine X and Y for M_k
        # X and Y are 2x2 matrices where each entry is a linear combination of A-blocks or B-blocks.
        # We represent an entry as a dict: {block_idx: coeff}
        
        # Initialize X and Y entries as empty dicts
        X = [[{}, {}, {}] for _ in range(2)] # 2x2
        Y = [[{}, {}, {}] for _ in range(2)] # 2x2
        
        # Fill X and Y based on k
        if k == 0:
            # M1 = (A00+A11)(B00+B11)
            # X = A00+A11, Y = B00+B11 (scalar blocks, so all 4 entries of X are A00+A11?)
            # No, X is a 2x2 matrix.
            # In Strassen, X and Y are the matrices being multiplied.
            # Here, the "matrices" are the blocks themselves?
            # Yes, M_k is the product of two 2x2 matrices formed by linear combinations of blocks.
            # So X is a 2x2 matrix where each entry is a block (or sum of blocks).
            
            # For M1:
            # X = [[A00, A00], [A11, A11]]? No.
            # The formula is (A00+A11)(B00+B11).
            # This implies X = A00+A11 (as a scalar multiple of identity? No.)
            # Wait, A00 and A11 are 2x2 matrices.
            # So X is the 2x2 matrix A00+A11.
            # So X[0][0] = A00[0][0] + A11[0][0], etc.
            # So for each entry (i,j) of X, the coefficient of block A00 is 1, and A11 is 1.
            
            # Let's define a function to add a block to an entry
            def add_block(entry_dict, block_idx, coeff):
                entry_dict[block_idx] = entry_dict.get(block_idx, 0) + coeff

            # M1
            for i in range(2):
                for j in range(2):
                    add_block(X[i][j], 0, 1) # A00
                    add_block(X[i][j], 3, 1) # A11
                    add_block(Y[i][j], 0, 1) # B00
                    add_block(Y[i][j], 3, 1) # B11
        elif k == 1:
            # M2 = (A10+A11)B00
            for i in range(2):
                for j in range(2):
                    add_block(X[i][j], 2, 1) # A10
                    add_block(X[i][j], 3, 1) # A11
                    add_block(Y[i][j], 0, 1) # B00
        elif k == 2:
            # M3 = A00(B01-B11)
            for i in range(2):
                for j in range(2):
                    add_block(X[i][j], 0, 1) # A00
                    add_block(Y[i][j], 1, 1) # B01
                    add_block(Y[i][j], 3, -1) # B11
        elif k == 3:
            # M4 = A11(B10-B00)
            for i in range(2):
                for j in range(2):
                    add_block(X[i][j], 3, 1) # A11
                    add_block(Y[i][j], 2, 1) # B10
                    add_block(Y[i][j], 0, -1) # B00
        elif k == 4:
            # M5 = (A00+A01)B11
            for i in range(2):
                for j in range(2):
                    add_block(X[i][j], 0, 1) # A00
                    add_block(X[i][j], 1, 1) # A01
                    add_block(Y[i][j], 3, 1) # B11
        elif k == 5:
            # M6 = (A10-A00)(B00+B01)
            for i in range(2):
                for j in range(2):
                    add_block(X[i][j], 2, 1) # A10
                    add_block(X[i][j], 0, -1) # A00
                    add_block(Y[i][j], 0, 1) # B00
                    add_block(Y[i][j], 1, 1) # B01
        elif k == 6:
            # M7 = (A01-A11)(B10+B11)
            for i in range(2):
                for j in range(2):
                    add_block(X[i][j], 1, 1) # A01
                    add_block(X[i][j], 3, -1) # A11
                    add_block(Y[i][j], 2, 1) # B10
                    add_block(Y[i][j], 3, 1) # B11

        # Now X and Y are 2x2 matrices of dicts.
        # We need to compute S_l for this X, Y.
        # S_l is a product of two linear combinations of the entries of X and Y.
        # Let u_x be the vector of coefficients for the first factor (from X entries).
        # Let v_y be the vector of coefficients for the second factor (from Y entries).
        
        # The entries of X are X[0][0], X[0][1], X[1][0], X[1][1].
        # Each X[i][j] is a dict of block coeffs.
        # The linear combination for the first factor of S_l is a sum of c * X[i][j].
        # So the total coefficient for a block b in the first factor is sum(c * X[i][j][b]).
        
        u_coeffs = {} # block_idx -> coeff
        v_coeffs = {} # block_idx -> coeff
        
        if l == 0:
            # S1 = (x00+x11)(y00+y11)
            # u: x00 + x11
            for b, c in X[0][0].items(): u_coeffs[b] = u_coeffs.get(b, 0) + c
            for b, c in X[1][1].items(): u_coeffs[b] = u_coeffs.get(b, 0) + c
            # v: y00 + y11
            for b, c in Y[0][0].items(): v_coeffs[b] = v_coeffs.get(b, 0) + c
            for b, c in Y[1][1].items(): v_coeffs[b] = v_coeffs.get(b, 0) + c
        elif l == 1:
            # S2 = (x10+x11)y00
            for b, c in X[1][0].items(): u_coeffs[b] = u_coeffs.get(b, 0) + c
            for b, c in X[1][1].items(): u_coeffs[b] = u_coeffs.get(b, 0) + c
            for b, c in Y[0][0].items(): v_coeffs[b] = v_coeffs.get(b, 0) + c
        elif l == 2:
            # S3 = x00(y01-y11)
            for b, c in X[0][0].items(): u_coeffs[b] = u_coeffs.get(b, 0) + c
            for b, c in Y[0][1].items(): v_coeffs[b] = v_coeffs.get(b, 0) + c
            for b, c in Y[1][1].items(): v_coeffs[b] = v_coeffs.get(b, 0) - c
        elif l == 3:
            # S4 = x11(y10-y00)
            for b, c in X[1][1].items(): u_coeffs[b] = u_coeffs.get(b, 0) + c
            for b, c in Y[1][0].items(): v_coeffs[b] = v_coeffs.get(b, 0) + c
            for b, c in Y[0][0].items(): v_coeffs[b] = v_coeffs.get(b, 0) - c
        elif l == 4:
            # S5 = (x00+x01)y11
            for b, c in X[0][0].items(): u_coeffs[b] = u_coeffs.get(b, 0) + c
            for b, c in X[0][1].items(): u_coeffs[b] = u_coeffs.get(b, 0) + c
            for b, c in Y[1][1].items(): v_coeffs[b] = v_coeffs.get(b, 0) + c
        elif l == 5:
            # S6 = (x10-x00)(y00+y01)
            for b, c in X[1][0].items(): u_coeffs[b] = u_coeffs.get(b, 0) + c
            for b, c in X[0][0].items(): u_coeffs[b] = u_coeffs.get(b, 0) - c
            for b, c in Y[0][0].items(): v_coeffs[b] = v_coeffs.get(b, 0) + c
            for b, c in Y[0][1].items(): v_coeffs[b] = v_coeffs.get(b, 0) + c
        elif l == 6:
            # S7 = (x01-x11)(y10+y11)
            for b, c in X[0][1].items(): u_coeffs[b] = u_coeffs.get(b, 0) + c
            for b, c in X[1][1].items(): u_coeffs[b] = u_coeffs.get(b, 0) - c
            for b, c in Y[1][0].items(): v_coeffs[b] = v_coeffs.get(b, 0) + c
            for b, c in Y[1][1].items(): v_coeffs[b] = v_coeffs.get(b, 0) + c

        # Now map u_coeffs (block coeffs) to global A indices
        for b, coeff in u_coeffs.items():
            if coeff == 0: continue
            indices = block_indices(b // 2, b % 2)
            for idx in indices:
                U[idx, c] += coeff
                
        # Map v_coeffs (block coeffs) to global B indices
        for b, coeff in v_coeffs.items():
            if coeff == 0: continue
            indices = block_indices(b // 2, b % 2)
            for idx in indices:
                V[idx, c] += coeff

        # Now determine the contribution to W
        # S_l contributes to M_k entries.
        # M_k entries contribute to C entries.
        
        # First, find which M_k entries S_l contributes to.
        m_contribs = {} # m_entry_idx (0..3) -> coeff
        
        if l == 0: # S1
            m_contribs[0] = 1 # z00
            m_contribs[3] = 1 # z11
        elif l == 1: # S2
            m_contribs[2] = 1 # z10
            m_contribs[3] = -1 # z11
        elif l == 2: # S3
            m_contribs[1] = 1 # z01
            m_contribs[3] = 1 # z11
        elif l == 3: # S4
            m_contribs[0] = 1 # z00
            m_contribs[2] = 1 # z10
        elif l == 4: # S5
            m_contribs[0] = -1 # z00
            m_contribs[1] = 1 # z01
        elif l == 5: # S6
            m_contribs[3] = 1 # z11
        elif l == 6: # S7
            m_contribs[0] = 1 # z00

        # Now map M_k entries to C entries.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M_k is a 2x2 matrix. Its entries are M_k[0][0], M_k[0][1], M_k[1][0], M_k[1][1].
        # These correspond to C blocks.
        # M_k[0][0] contributes to C00
        # M_k[0][1] contributes to C01
        # M_k[1][0] contributes to C10
        # M_k[1][1] contributes to C11
        
        c_contribs = {} # c_block_idx (0..3) -> coeff
        
        for m_entry, m_coeff in m_contribs.items():
            # m_entry 0 -> C00, 1 -> C01, 2 -> C10, 3 -> C11
            # We need to check the outer reconstruction formula for M_k.
            
            # Let's define the outer coefficients for M_k in C.
            # C00: M1(1), M4(1), M5(-1), M7(1)
            # C01: M3(1), M5(1)
            # C10: M2(1), M4(1)
            # C11: M1(1), M2(-1), M3(1), M6(1)
            
            # So for a given k, what is the coefficient of M_k in each C block?
            outer_coeff = {}
            if k == 0: # M1
                outer_coeff[0] = 1 # C00
                outer_coeff[3] = 1 # C11
            elif k == 1: # M2
                outer_coeff[2] = 1 # C10
                outer_coeff[3] = -1 # C11
            elif k == 2: # M3
                outer_coeff[1] = 1 # C01
                outer_coeff[3] = 1 # C11
            elif k == 3: # M4
                outer_coeff[0] = 1 # C00
                outer_coeff[2] = 1 # C10
            elif k == 4: # M5
                outer_coeff[0] = -1 # C00
                outer_coeff[1] = 1 # C01
            elif k == 5: # M6
                outer_coeff[3] = 1 # C11
            elif k == 6: # M7
                outer_coeff[0] = 1 # C00

            for c_block, c_coeff in outer_coeff.items():
                # m_entry corresponds to an entry of M_k.
                # Does m_entry affect which C block it goes to?
                # Yes, M_k is a 2x2 matrix.
                # M_k[0][0] goes to C00.
                # M_k[0][1] goes to C01.
                # M_k[1][0] goes to C10.
                # M_k[1][1] goes to C11.
                
                # So the target C block is determined by m_entry.
                # But the coefficient from the outer formula depends on k and the target C block.
                # Wait, the outer formula says C00 = M1 + ...
                # This means the entire matrix M1 contributes to C00?
                # No, C00 is a 2x2 block. M1 is a 2x2 matrix.
                # So M1[0][0] contributes to C00[0][0], M1[0][1] to C00[0][1], etc.
                # So the coefficient is the same for all entries of M1 when adding to C00.
                
                # So, if m_entry is 0 (M_k[0][0]), it contributes to C block 0.
                # If m_entry is 1 (M_k[0][1]), it contributes to C block 1.
                # If m_entry is 2 (M_k[1][0]), it contributes to C block 2.
                # If m_entry is 3 (M_k[1][1]), it contributes to C block 3.
                
                target_c_block = m_entry
                
                # Get the coefficient of M_k in this target_c_block from outer_coeff
                coeff = outer_coeff.get(target_c_block, 0)
                
                if coeff != 0:
                    total_coeff = m_coeff * coeff
                    c_contribs[target_c_block] = c_contribs.get(target_c_block, 0) + total_coeff

        # Now map c_contribs to global C indices
        for c_block, coeff in c_contribs.items():
            if coeff == 0: continue
            indices = block_indices(c_block // 2, c_block % 2)
            for idx in indices:
                W[idx, c] += coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
