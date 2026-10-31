import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    rank = 49
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Index conventions:
    # U: A is n×m, flattened row-major: index = i*m + k
    # V: B is m×p, flattened row-major: index = k*p + j  
    # W: C is n×p, "official trace convention W[k*n+i,r]" means W[j*n+i, t] for C[i,j]
    # So W is flattened column-major for C: index = j*n + i (col*n + row)
    
    def a_idx(i, k):
        return i * m + k
        
    def b_idx(k, j):
        return k * p + j
        
    def w_idx(i, j):
        return j * n + i  # column-major: col*n + row
    
    # Recursive Strassen: 4x4 = 2x2 of 2x2 blocks
    # Level 1: 7 multipliers, each involving 2x2 sub-blocks
    # Level 2: each 2x2 multiplication uses 7 more multipliers (base case for 2x2 entries)
    
    # Actually, for rank 49 = 7 * 7, we apply Strassen twice:
    # First level: 7 multipliers on 4x4, each giving products of 2x2 matrices
    # Second level: each 2x2 matrix multiplication uses 7 Strassen multipliers
    
    # The 7 Strassen formulas for 2x2:
    # M1 = (A00+A11)(B00+B11), C00 += M1, C11 += M1
    # M2 = (A10+A11)B00,       C10 += M2, C11 -= M2
    # M3 = A00(B01-B11),       C01 += M3, C11 += M3
    # M4 = A11(B10-B00),       C00 += M4, C10 += M4
    # M5 = (A00+A01)B11,       C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01), C11 += M6
    # M7 = (A01-A11)(B10+B11), C00 += M7
    
    # For the tensor decomposition, we work with linear combinations.
    # Each "entry" at level 1 is itself a 2x2 matrix (4 scalar entries).
    # So we need to track coefficients for all 4x4 = 16 scalar positions.
    
    # Represent a 4x4 matrix as dict: (global_row, global_col) -> coeff
    # Or as 2x2 blocks: (br, bc, lr, lc) -> coeff where br,bc in {0,1}, lr,lc in {0,1}
    
    def combine_blocks(block1, block2, sign1=1.0, sign2=1.0):
        """
        block1, block2: dict (br, bc, lr, lc) -> coeff
        Returns new dict with combined coefficients.
        """
        result = {}
        for key in set(list(block1.keys()) + list(block2.keys())):
            c = 0.0
            if key in block1:
                c += sign1 * block1[key]
            if key in block2:
                c += sign2 * block2[key]
            if abs(c) > 1e-15:
                result[key] = c
        return result
    
    # Level 1: identify the 7 multipliers
    # Each multiplier X_t, Y_t are 2x2 block combinations
    # X_t: dict (br, bc) -> list of (lr, lc, coeff) but easier: (br, bc, lr, lc) -> coeff
    
    # M1: X = A00+A11, Y = B00+B11
    X1 = {}
    for br, bc in [(0,0), (1,1)]:
        for lr in range(2):
            for lc in range(2):
                X1[(br, bc, lr, lc)] = 1.0
    Y1 = {}
    for br, bc in [(0,0), (1,1)]:
        for lr in range(2):
            for lc in range(2):
                Y1[(br, bc, lr, lc)] = 1.0
    # C contributions: C00 += 1, C11 += 1
    C1 = {}
    for br, bc in [(0,0), (1,1)]:
        for lr in range(2):
            for lc in range(2):
                C1[(br, bc, lr, lc)] = 1.0
    
    # M2: X = A10+A11, Y = B00
    X2 = {}
    for br, bc in [(1,0), (1,1)]:
        for lr in range(2):
            for lc in range(2):
                X2[(br, bc, lr, lc)] = 1.0
    Y2 = {}
    for br, bc in [(0,0)]:
        for lr in range(2):
            for lc in range(2):
                Y2[(br, bc, lr, lc)] = 1.0
    C2 = {}
    for br, bc in [(1,0)]:
        for lr in range(2):
            for lc in range(2):
                C2[(br, bc, lr, lc)] = 1.0
    for br, bc in [(1,1)]:
        for lr in range(2):
            for lc in range(2):
                C2[(br, bc, lr, lc)] = -1.0
    
    # M3: X = A00, Y = B01-B11
    X3 = {}
    for lr in range(2):
        for lc in range(2):
            X3[(0, 0, lr, lc)] = 1.0
    Y3 = {}
    for lr in range(2):
        for lc in range(2):
            Y3[(0, 1, lr, lc)] = 1.0
            Y3[(1, 1, lr, lc)] = -1.0
    C3 = {}
    for br, bc in [(0,1), (1,1)]:
        for lr in range(2):
            for lc in range(2):
                C3[(br, bc, lr, lc)] = 1.0
    
    # M4: X = A11, Y = B10-B00
    X4 = {}
    for lr in range(2):
        for lc in range(2):
            X4[(1, 1, lr, lc)] = 1.0
    Y4 = {}
    for lr in range(2):
        for lc in range(2):
            Y4[(1, 0, lr, lc)] = 1.0
            Y4[(0, 0, lr, lc)] = -1.0
    C4 = {}
    for br, bc in [(0,0), (1,0)]:
        for lr in range(2):
            for lc in range(2):
                C4[(br, bc, lr, lc)] = 1.0
    
    # M5: X = A00+A01, Y = B11
    X5 = {}
    for br, bc in [(0,0), (0,1)]:
        for lr in range(2):
            for lc in range(2):
                X5[(br, bc, lr, lc)] = 1.0
    Y5 = {}
    for lr in range(2):
        for lc in range(2):
            Y5[(1, 1, lr, lc)] = 1.0
    C5 = {}
    for br, bc in [(0,0)]:
        for lr in range(2):
            for lc in range(2):
                C5[(br, bc, lr, lc)] = -1.0
    for br, bc in [(0,1)]:
        for lr in range(2):
            for lc in range(2):
                C5[(br, bc, lr, lc)] = 1.0
    
    # M6: X = A10-A00, Y = B00+B01
    X6 = {}
    for lr in range(2):
        for lc in range(2):
            X6[(1, 0, lr, lc)] = 1.0
            X6[(0, 0, lr, lc)] = -1.0
    Y6 = {}
    for br, bc in [(0,0), (0,1)]:
        for lr in range(2):
            for lc in range(2):
                Y6[(br, bc, lr, lc)] = 1.0
    C6 = {}
    for lr in range(2):
        for lc in range(2):
            C6[(1, 1, lr, lc)] = 1.0
    
    # M7: X = A01-A11, Y = B10+B11
    X7 = {}
    for lr in range(2):
        for lc in range(2):
            X7[(0, 1, lr, lc)] = 1.0
            X7[(1, 1, lr, lc)] = -1.0
    Y7 = {}
    for br, bc in [(1,0), (1,1)]:
        for lr in range(2):
            for lc in range(2):
                Y7[(br, bc, lr, lc)] = 1.0
    C7 = {}
    for lr in range(2):
        for lc in range(2):
            C7[(0, 0, lr, lc)] = 1.0
    
    level1 = [(X1,Y1,C1), (X2,Y2,C2), (X3,Y3,C3), (X4,Y4,C4), (X5,Y5,C5), (X6,Y6,C6), (X7,Y7,C7)]
    
    # Now for each level-1 multiplier, apply level-2 Strassen
    # The X, Y at level 1 are linear combinations of 2x2 blocks.
    # To multiply them, we treat each as a 2x2 matrix where entries are themselves linear combos of scalar positions.
    # We need 7 more multipliers per level-1 term.
    
    def get_local_block(block_dict, br, bc):
        """Get the 2x2 local entries for block (br,bc) from block_dict"""
        result = {}
        for (b_r, b_c, lr, lc), coeff in block_dict.items():
            if b_r == br and b_c == bc:
                result[(lr, lc)] = coeff
        return result
    
    def combine_local(block, lr, lc, coeff_mult=1.0):
        """
        block: dict (br, bc) -> dict (lr, lc) -> coeff
        Returns new dict (br, bc, lr, lc) -> coeff for a single position (lr,lc) scaled.
        """
        result = {}
        for (br, bc), local in block.items():
            if (lr, lc) in local:
                c = coeff_mult * local[(lr, lc)]
                if abs(c) > 1e-15:
                    result[(br, bc, lr, lc)] = c
        return result
    
    col = 0
    for (X_l1, Y_l1, C_l1) in level1:
        # Convert to nested: dict (br,bc) -> dict (lr,lc) -> coeff
        def to_nested(flat_dict):
            nested = {}
            for (br, bc, lr, lc), coeff in flat_dict.items():
                if (br, bc) not in nested:
                    nested[(br, bc)] = {}
                nested[(br, bc)][(lr, lc)] = nested[(br, bc)].get((lr, lc), 0) + coeff
            return nested
        
        X_nested = to_nested(X_l1)
        Y_nested = to_nested(Y_l1)
        
        # Level 2: 7 Strassen multipliers for 2x2 "matrices"
        # Each entry is a dict (lr,lc) -> coeff (local 2x2 entries)
        
        X00 = X_nested.get((0,0), {})
        X01 = X_nested.get((0,1), {})
        X10 = X_nested.get((1,0), {})
        X11 = X_nested.get((1,1), {})
        
        Y00 = Y_nested.get((0,0), {})
        Y01 = Y_nested.get((0,1), {})
        Y10 = Y_nested.get((1,0), {})
        Y11 = Y_nested.get((1,1), {})
        
        # Helper to merge local dicts
        def merge_local(d1, d2, sign1=1.0, sign2=1.0):
            result = {}
            for key in set(list(d1.keys()) + list(d2.keys())):
                c = 0.0
                if key in d1:
                    c += sign1 * d1[key]
                if key in d2:
                    c += sign2 * d2[key]
                if abs(c) > 1e-15:
                    result[key] = c
            return result
        
        # Apply 7 level-2 multipliers
        # M2_1: (X00+X11)(Y00+Y11) -> C_local00 += 1, C_local11 += 1
        # M2_2: (X10+X11)Y00       -> C_local10 += 1, C_local11 -= 1
        # M2_3: X00(Y01-Y11)       -> C_local01 += 1, C_local11 += 1
        # M2_4: X11(Y10-Y00)       -> C_local00 += 1, C_local10 += 1
        # M2_5: (X00+X01)Y11       -> C_local00 -= 1, C_local01 += 1
        # M2_6: (X10-X00)(Y00+Y01) -> C_local11 += 1
        # M2_7: (X01-X11)(Y10+Y11) -> C_local00 += 1
        
        # For each, the result contributes to C_l1 positions
        # C_l1 tells us which 4x4 C-blocks are affected and with what coefficient.
        # The local 2x2 result (lr,lc) maps to global C position (br*2+lr, bc*2+lc)
        
        # Level-2 multiplier definitions: (X_combos, Y_combos, C_local_targets)
        # C_local_targets: dict (c_br, c_bc, lr, lc) -> coeff where c_br,c_bc in {0,1} for C 2x2 block
        # and lr,lc are local positions within that block
        
        l2_multipliers = [
            # M2_1
            (merge_local(X00, X11), merge_local(Y00, Y11), 
             {(0,0): {(lr,lc): 1.0 for lr in range(2) for lc in range(2)},
               (1,1): {(lr,lc): 1.0 for lr in range(2) for lc in range(2)}}),
            # M2_2
            (merge_local(X10, X11), Y00,
             {(1,0): {(lr,lc): 1.0 for lr in range(2) for lc in range(2)},
               (1,1): {(lr,lc): -1.0 for lr in range(2) for lc in range(2)}}),
            # M2_3
            (X00, merge_local(Y01, Y11, 1.0, -1.0),
             {(0,1): {(lr,lc): 1.0 for lr in range(2) for lc in range(2)},
               (1,1): {(lr,lc): 1.0 for lr in range(2) for lc in range(2)}}),
            # M2_4
            (X11, merge_local(Y10, Y00, 1.0, -1.0),
             {(0,0): {(lr,lc): 1.0 for lr in range(2) for lc in range(2)},
               (1,0): {(lr,lc): 1.0 for lr in range(2) for lc in range(2)}}),
            # M2_5
            (merge_local(X00, X01), Y11,
             {(0,0): {(lr,lc): -1.0 for lr in range(2) for lc in range(2)},
               (0,1): {(lr,lc): 1.0 for lr in range(2) for lc in range(2)}}),
            # M2_6
            (merge_local(X10, X00, 1.0, -1.0), merge_local(Y00, Y01),
             {(1,1): {(lr,lc): 1.0 for lr in range(2) for lc in range(2)}}),
            # M2_7
            (merge_local(X01, X11, 1.0, -1.0), merge_local(Y10, Y11),
             {(0,0): {(lr,lc): 1.0 for lr in range(2) for lc in range(2)}}),
        ]
        
        for (X_l2, Y_l2, C_l2_targets) in l2_multipliers:
            U_col = np.zeros(n*m)
            V_col = np.zeros(m*p)
            W_col = np.zeros(p*n)
            
            # Fill U from X_l2: dict (br,bc) -> dict (lr,lc) -> coeff
            for (br, bc), local in X_l2.items():
                for (lr, lc), coeff in local.items():
                    gi = br * 2 + lr
                    gj = bc * 2 + lc
                    U_col[a_idx(gi, gj)] += coeff
            
            # Fill V from Y_l2
            for (br, bc), local in Y_l2.items():
                for (lr, lc), coeff in local.items():
                    gk = br * 2 + lr
                    gj = bc * 2 + lc
                    V_col[b_idx(gk, gj)] += coeff
            
            # Fill W from C_l2_targets, modulated by C_l1
            # C_l2_targets: dict (c_br, c_bc) -> dict (lr, lc) -> coeff
            # These are local 2x2 results. Map to global C positions.
            # The 2x2 C-block (c_br, c_bc) at local (lr,lc) maps to global (c_br*2+lr, c_bc*2+lc)
            # But we also need to apply the C_l1 coefficients which tell us
            # which level-1 C-blocks this contributes to and with what sign.
            
            # C_l1 is a flat dict (br, bc, lr, lc) -> coeff for the 4x4 C matrix.
            # The level-2 result computes a 2x2 matrix Z.
            # Z[lr, lc] contributes to C positions defined by C_l1.
            # C_l1 entries have (br, bc, lr, lc) where br,bc select the 2x2 C-block
            # and lr,lc are local positions.
            
            # Actually, C_l1 already encodes the full 4x4 contribution pattern.
            # The level-2 multiplication gives a 2x2 result. Each entry Z[lr2, lc2] 
            # (where lr2,lc2 in {0,1}) should be multiplied by the corresponding
            # C_l1 coefficients.
            
            # Let me reconsider: C_l1 has entries like (0,0,0,0)->1 meaning C[0,0] gets +1 from this multiplier.
            # The level-2 2x2 result Z has entries Z[0,0], Z[0,1], Z[1,0], Z[1,1].
            # Z[0,0] contributes to all C_l1 entries where local pos is (0,0).
            # Z[0,1] contributes to C_l1 entries where local pos is (0,1).
            # etc.
            
            for (c_br, c_bc), c_local_dict in C_l2_targets.items():
                for (lr2, lc2), z_coeff in c_local_dict.items():
                    # Find all C_l1 entries where (br, bc) = (c_br, c_bc) and (lr, lc) = (lr2, lc2)
                    for (b_r, b_c, l_r, l_c), c1_coeff in C_l1.items():
                        if b_r == c_br and b_c == c_bc and l_r == lr2 and l_c == lc2:
                            # Global C position: (b_r*2 + l_r, b_c*2 + l_c)
                            g_i = b_r * 2 + l_r
                            g_j = b_c * 2 + l_c
                            W_col[w_idx(g_i, g_j)] += z_coeff * c1_coeff
            
            U[:, col] = U_col
            V[:, col] = V_col
            W[:, col] = W_col
            col += 1
            
    assert col == rank, f"Expected {rank}, got {col}"
    
    # Verify tensor decomposition
    for i in range(n):
        for j in range(p):
            for k in range(m):
                val = 0.0
                for t in range(rank):
                    val += U[i*m+k, t] * V[k*p+j, t] * W[j*n+i, t]
                if abs(val - 1.0) > 1e-10:
                    raise ValueError(f"Tensor check failed at ({i},{j},{k}): {val}")
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
