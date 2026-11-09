import time
import sys

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    
    # We will build U, V, W as lists of lists.
    # U: (n*m, R) = (20, R)
    # V: (m*p, R) = (25, R)
    # W: (p*n, R) = (20, R)
    
    terms = []
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val):
        terms.append((u_idx, v_idx, w_idx, val))
        
    # 1. Strassen for 4x4x4 (A_L B_TL)
    # A_L is rows 0-3, cols 0-3 of A.
    # B_TL is rows 0-3, cols 0-3 of B.
    # C is rows 0-3, cols 0-3 of C.
    
    # We need to generate the 49 terms.
    # We can do this by iterating over the 7 Strassen steps at level 2,
    # and for each, 7 Strassen steps at level 1.
    
    # Strassen 2x2 decomposition
    # Returns list of (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: 4 floats (for A 2x2)
    # v_coeffs: 4 floats (for B 2x2)
    # w_coeffs: 4 floats (for C 2x2)
    
    def strassen_2x2_terms():
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M5 + M3 + M4
        # C01 = M5 + M6
        # C10 = M2 + M7
        # C11 = M1 - M2 + M4 - M7
        
        # We represent each M as a term in the decomposition.
        # The output C is a linear combination of M's.
        # So for each M, we have a U part, V part, and W part (which is the coefficient in C).
        
        res = []
        
        # M1
        u = [1, 0, 1, 0] # A00, A01, A10, A11 -> A00+A11
        v = [1, 1, 1, 1] # B00, B01, B10, B11 -> B00+B11
        # C coeffs: C00+=1, C11+=1
        w = [1, 0, 0, 1]
        res.append((u, v, w))
        
        # M2
        u = [0, 0, 1, 1] # A10+A11
        v = [1, 0, 0, 0] # B00
        w = [0, 0, 1, 0] # C10 += 1
        res.append((u, v, w))
        
        # M3
        u = [1, 0, 0, 0] # A00
        v = [0, 1, 0, -1] # B01-B11
        w = [1, 0, 0, 0] # C00 += 1
        res.append((u, v, w))
        
        # M4
        u = [0, 0, 0, 1] # A11
        v = [0, 0, 1, -1] # B10-B00
        w = [1, 0, 0, 0] # C00 += 1
        res.append((u, v, w))
        
        # M5
        u = [1, 1, 0, 0] # A00+A01
        v = [0, 0, 0, 1] # B11
        w = [0, 1, 0, 0] # C01 += 1
        res.append((u, v, w))
        
        # M6
        u = [0, 0, 1, -1] # A10-A00
        v = [1, 1, 0, 0] # B00+B01
        w = [0, 1, 0, 0] # C01 += 1
        res.append((u, v, w))
        
        # M7
        u = [0, 1, 0, -1] # A01-A11
        v = [0, 0, 1, 1] # B10+B11
        w = [0, 0, 1, 0] # C10 += 1
        res.append((u, v, w))
        
        return res

    s2_terms = strassen_2x2_terms()
    
    # Now compose 4x4 from 2x2
    # A 4x4 is 4 blocks of 2x2: A00, A01, A10, A11
    # B 4x4 is 4 blocks: B00, B01, B10, B11
    # C 4x4 is 4 blocks: C00, C01, C10, C11
    
    # We apply strassen_2x2 to the blocks.
    # Each term in s2_terms has u, v, w of length 4 (block coeffs).
    # For each such term, we expand into 7 scalar terms by applying s2_terms again to the blocks.
    
    # Actually, we can just iterate.
    # For each term (u_b, v_b, w_b) in s2_terms:
    #   For each sub-term (u_s, v_s, w_s) in s2_terms:
    #     The u part is a 4x4 matrix of coeffs.
    #     u_s gives coeffs for the 2x2 block.
    #     u_b gives which block.
    
    # Let's build the 49 terms for 4x4x4.
    # Each term will have:
    # U: 16 coeffs (for 4x4 A)
    # V: 16 coeffs (for 4x4 B)
    # W: 16 coeffs (for 4x4 C)
    
    # We will generate these 16-length vectors.
    
    def expand_4x4():
        terms_4x4 = []
        for u_b, v_b, w_b in s2_terms:
            for u_s, v_s, w_s in s2_terms:
                # u_b is 4 coeffs for blocks 0,1,2,3
                # u_s is 4 coeffs for elements in a block
                # We need to map to 16 elements.
                
                u_full = [0]*16
                v_full = [0]*16
                w_full = [0]*16
                
                # Block 0: rows 0-1, cols 0-1
                # Block 1: rows 0-1, cols 2-3
                # Block 2: rows 2-3, cols 0-1
                # Block 3: rows 2-3, cols 2-3
                
                for b_idx in range(4):
                    # u_b[b_idx] is the weight for this block
                    # u_s gives the internal structure
                    # We need to add u_b[b_idx] * u_s to u_full at the correct positions.
                    
                    # Positions for block b_idx in 4x4 matrix (row-major? or block-major?)
                    # Let's use row-major for the 4x4 matrix indices 0..15.
                    # Block 0: 0,1,4,5
                    # Block 1: 2,3,6,7
                    # Block 2: 8,9,12,13
                    # Block 3: 10,11,14,15
                    
                    # Wait, standard 2x2 block layout in 4x4:
                    # A00 A01
                    # A10 A11
                    # A00 is top-left 2x2.
                    
                    # Indices in 4x4 (row-major):
                    # Row 0: 0,1,2,3
                    # Row 1: 4,5,6,7
                    # Row 2: 8,9,10,11
                    # Row 3: 12,13,14,15
                    
                    # Block 0 (A00): 0,1,4,5
                    # Block 1 (A01): 2,3,6,7
                    # Block 2 (A10): 8,9,12,13
                    # Block 3 (A11): 10,11,14,15
                    
                    # u_s indices: 0,1,2,3 correspond to A00,A01,A10,A11 of the 2x2 block.
                    # So u_s[0] -> pos 0, u_s[1] -> pos 1, u_s[2] -> pos 4, u_s[3] -> pos 5 for Block 0.
                    
                    pos_map = [
                        [0, 1, 4, 5], # Block 0
                        [2, 3, 6, 7], # Block 1
                        [8, 9, 12, 13], # Block 2
                        [10, 11, 14, 15] # Block 3
                    ]
                    
                    # For U
                    for k in range(4):
                        val = u_b[b_idx] * u_s[k]
                        if val != 0:
                            u_full[pos_map[b_idx][k]] += val
                            
                    # For V
                    for k in range(4):
                        val = v_b[b_idx] * v_s[k]
                        if val != 0:
                            v_full[pos_map[b_idx][k]] += val
                            
                    # For W
                    for k in range(4):
                        val = w_b[b_idx] * w_s[k]
                        if val != 0:
                            w_full[pos_map[b_idx][k]] += val
                            
                terms_4x4.append((u_full, v_full, w_full))
        return terms_4x4

    t_4x4 = expand_4x4()
    
    # Map these to the global indices for Term 1
    # A_L: rows 0-3, cols 0-3.
    # Global A index: r*5 + c.
    # Local A index (4x4): r*4 + c.
    
    # B_TL: rows 0-3, cols 0-3.
    # Global B index: r*5 + c.
    # Local B index (4x4): r*4 + c.
    
    # C: rows 0-3, cols 0-3.
    # Global C index: r*5 + c.
    # W index: k*n + i = c*4 + r.
    
    for u_full, v_full, w_full in t_4x4:
        # u_full is 16 coeffs for A_L (4x4)
        # We need to place them in U (20xR) at the correct columns for this term.
        # But we are building terms list first.
        
        # Convert local 4x4 index to global A index
        u_global = [0]*20
        for local_idx, val in enumerate(u_full):
            if val != 0:
                r = local_idx // 4
                c = local_idx % 4
                global_idx = r * 5 + c
                u_global[global_idx] = val
                
        v_global = [0]*25
        for local_idx, val in enumerate(v_full):
            if val != 0:
                r = local_idx // 4
                c = local_idx % 4
                global_idx = r * 5 + c
                v_global[global_idx] = val
                
        w_global = [0]*20
        for local_idx, val in enumerate(w_full):
            if val != 0:
                # w_full is for C 4x4.
                # C local index is row-major r*4+c?
                # My pos_map for W was same as A/B.
                # So local_idx corresponds to r*4+c in C.
                r = local_idx // 4
                c = local_idx % 4
                # W index is k*n + i = c*4 + r
                global_idx = c * 4 + r
                w_global[global_idx] = val
                
        terms.append((u_global, v_global, w_global))

    # 2. Schoolbook for Term 2: A_L B_TR (4x4x1)
    # A_L: rows 0-3, cols 0-3.
    # B_TR: rows 0-3, col 4.
    # C: rows 0-3, col 4.
    
    for r in range(4):
        for c in range(4):
            u_g = [0]*20
            u_g[r*5 + c] = 1
            
            v_g = [0]*25
            v_g[c*5 + 4] = 1 # B row c, col 4
            
            w_g = [0]*20
            # C row r, col 4.
            # W index: k*n + i = 4*4 + r = 16 + r
            w_g[16 + r] = 1
            
            terms.append((u_g, v_g, w_g))

    # 3. Schoolbook for Term 3: A_R B_B (4x1x5)
    # A_R: rows 0-3, col 4.
    # B_B: row 4, cols 0-4.
    # C: rows 0-3, cols 0-4.
    
    for r in range(4):
        for c in range(5):
            u_g = [0]*20
            u_g[r*5 + 4] = 1 # A row r, col 4
            
            v_g = [0]*25
            v_g[4*5 + c] = 1 # B row 4, col c
            
            w_g = [0]*20
            # C row r, col c.
            # W index: c*4 + r
            w_g[c*4 + r] = 1
            
            terms.append((u_g, v_g, w_g))

    # Build U, V, W
    R = len(terms)
    U = [[0]*R for _ in range(20)]
    V = [[0]*R for _ in range(25)]
    W = [[0]*R for _ in range(20)]
    
    for k, (u_g, v_g, w_g) in enumerate(terms):
        for i in range(20):
            if u_g[i] != 0:
                U[i][k] = u_g[i]
        for i in range(25):
            if v_g[i] != 0:
                V[i][k] = v_g[i]
        for i in range(20):
            if w_g[i] != 0:
                W[i][k] = w_g[i]
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
