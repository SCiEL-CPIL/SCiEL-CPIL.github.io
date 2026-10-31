"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Returns a rank-49 tensor decomposition for 4x4 matrix multiplication.
    This uses the recursive Strassen algorithm (7 outer Strassen steps, 
    each using 7 inner Strassen steps, total 49 multiplications).
    """
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a linear combination of A entries to U
    # A entries are indexed by (row, col). Flat index is row * m + col.
    # u_coeffs: list of (row, col, coeff)
    def add_u(col_idx, u_coeffs):
        for r, c, coeff in u_coeffs:
            U[r * m + c, col_idx] += coeff
            
    # Helper to add a linear combination of B entries to V
    # B entries are indexed by (row, col). Flat index is row * p + col.
    # v_coeffs: list of (row, col, coeff)
    def add_v(col_idx, v_coeffs):
        for r, c, coeff in v_coeffs:
            V[r * p + c, col_idx] += coeff
            
    # Helper to add a linear combination of C entries to W
    # C entries are indexed by (row, col). Flat index is col * n + row.
    # w_coeffs: list of (row, col, coeff)
    def add_w(col_idx, w_coeffs):
        for r, c, coeff in w_coeffs:
            W[c * n + r, col_idx] += coeff

    # Define the 7 outer Strassen terms for 4x4 (blocks of 2x2)
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # Each Aij, Bij is 2x2.
    
    # We represent linear combinations of A entries as lists of (row, col, coeff)
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    def get_a_block(br, bc):
        # br, bc in {0, 1}
        rows = [br * 2, br * 2 + 1]
        cols = [bc * 2, bc * 2 + 1]
        return [(r, c, 1) for r in rows for c in cols]
        
    def get_b_block(br, bc):
        # br, bc in {0, 1}
        rows = [br * 2, br * 2 + 1]
        cols = [bc * 2, bc * 2 + 1]
        return [(r, c, 1) for r in rows for c in cols]

    A00 = get_a_block(0, 0); A01 = get_a_block(0, 1)
    A10 = get_a_block(1, 0); A11 = get_a_block(1, 1)
    
    B00 = get_b_block(0, 0); B01 = get_b_block(0, 1)
    B10 = get_b_block(1, 0); B11 = get_b_block(1, 1)

    # Outer Strassen Terms X, Y
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # c_targets[t] defines how the result of outer term t contributes to C blocks
    # Each entry is (block_row, block_col, coeff)
    # Block indices: (0,0)=C00, (0,1)=C01, (1,0)=C10, (1,1)=C11
    c_targets = [
        [(0,0,1), (1,1,1)],       # M1
        [(1,0,1), (1,1,-1)],      # M2
        [(0,1,1), (1,1,1)],       # M3
        [(0,0,1), (1,0,1)],       # M4
        [(0,0,-1), (0,1,1)],      # M5
        [(1,1,1)],                 # M6
        [(0,0,1)],                 # M7
    ]

    outer_terms = [
        (A00 + A11, B00 + B11), # M1
        (A10 + A11, B00),       # M2
        (A00, B01 + [(r, c, -1) for r, c, _ in B11]), # M3
        (A11, B10 + [(r, c, -1) for r, c, _ in B00]), # M4
        (A00 + A01, B11),       # M5
        (A10 + [(r, c, -1) for r, c, _ in A00], B00 + B01), # M6
        (A01 + [(r, c, -1) for r, c, _ in A11], B10 + B11), # M7
    ]

    col = 0
    for t in range(7):
        X_coeffs = outer_terms[t][0]
        Y_coeffs = outer_terms[t][1]
        
        # Inner Strassen on X (2x2) and Y (2x2)
        # X is a 2x2 matrix of linear combos of A entries.
        # We need to extract the 4 blocks of X: X00, X01, X10, X11.
        # X_coeffs is a flat list of (r, c, coeff) for the 4x4 matrix A?
        # No, X is a 2x2 matrix where each entry is a 2x2 matrix of A entries.
        # So X[0,0] corresponds to A00, X[0,1] to A01, etc.
        # But X_coeffs is a combination of these blocks.
        # We need to decompose X_coeffs into 4 groups: X00, X01, X10, X11.
        
        # Helper to split a list of (r,c,v) into 4 blocks based on r,c
        def split_blocks(coeffs):
            X00 = []; X01 = []; X10 = []; X11 = []
            for r, c, v in coeffs:
                br = r // 2
                bc = c // 2
                if br == 0 and bc == 0:
                    X00.append((r, c, v))
                elif br == 0 and bc == 1:
                    X01.append((r, c, v))
                elif br == 1 and bc == 0:
                    X10.append((r, c, v))
                elif br == 1 and bc == 1:
                    X11.append((r, c, v))
            return X00, X01, X10, X11

        X00, X01, X10, X11 = split_blocks(X_coeffs)
        Y00, Y01, Y10, Y11 = split_blocks(Y_coeffs)

        # Inner Strassen Terms for X, Y
        # M1_in = (X00+X11)(Y00+Y11)
        # M2_in = (X10+X11)Y00
        # M3_in = X00(Y01-Y11)
        # M4_in = X11(Y10-Y00)
        # M5_in = (X00+X01)Y11
        # M6_in = (X10-X00)(Y00+Y01)
        # M7_in = (X01-X11)(Y10+Y11)
        
        # C00_in = M1_in + M4_in - M5_in + M7_in
        # C01_in = M3_in + M5_in
        # C10_in = M2_in + M4_in
        # C11_in = M1_in - M2_in + M3_in + M6_in
        
        # c_targets_in[s] defines how inner term s contributes to C blocks
        # Block indices: (0,0)=C00, (0,1)=C01, (1,0)=C10, (1,1)=C11
        # Note: These are local 2x2 blocks.
        c_targets_in = [
            [(0,0,1), (1,1,1)],       # M1_in
            [(1,0,1), (1,1,-1)],      # M2_in
            [(0,1,1), (1,1,1)],       # M3_in
            [(0,0,1), (1,0,1)],       # M4_in
            [(0,0,-1), (0,1,1)],      # M5_in
            [(1,1,1)],                 # M6_in
            [(0,0,1)],                 # M7_in
        ]

        inner_terms = [
            (X00 + X11, Y00 + Y11), # M1_in
            (X10 + X11, Y00),       # M2_in
            (X00, Y01 + [(r, c, -1) for r, c, _ in Y11]), # M3_in
            (X11, Y10 + [(r, c, -1) for r, c, _ in Y00]), # M4_in
            (X00 + X01, Y11),       # M5_in
            (X10 + [(r, c, -1) for r, c, _ in X00], Y00 + Y01), # M6_in
            (X01 + [(r, c, -1) for r, c, _ in X11], Y10 + Y11), # M7_in
        ]

        for s in range(7):
            u_coeffs = inner_terms[s][0]
            v_coeffs = inner_terms[s][1]
            
            add_u(col, u_coeffs)
            add_v(col, v_coeffs)
            
            # Map local C block (br, bc) and local (lr, lc) to global W index
            # W index is col * n + row
            # Global C block (br, bc) covers rows br*2 to br*2+1, cols bc*2 to bc*2+1
            # Inner term s contributes to local C block (lbr, lbc) with coeff c_val
            # The local C block (lbr, lbc) is a 2x2 matrix.
            # We need to add to W for each entry in this 2x2 block.
            
            for (lbr, lbc, c_val) in c_targets_in[s]:
                # Global block row/col
                gbr = lbr
                gbc = lbc
                
                # Wait, the inner Strassen operates on the 2x2 blocks X, Y.
                # The result of the inner Strassen is a 2x2 matrix of scalars (products).
                # These products sum up to the 2x2 block C_local.
                # C_local[0,0] = M1_in + M4_in - M5_in + M7_in
                # C_local[0,1] = M3_in + M5_in
                # C_local[1,0] = M2_in + M4_in
                # C_local[1,1] = M1_in - M2_in + M3_in + M6_in
                
                # So for each inner term s, we know which local C entries it contributes to.
                # c_targets_in[s] gives (lbr, lbc, coeff).
                # This means term s contributes `coeff` * (product) to C_local[lbr, lbc].
                
                # The product is stored in column `col` of U and V.
                # We need to add `coeff` to W for the corresponding global C entry.
                
                # Global row = gbr * 2 + lr
                # Global col = gbc * 2 + lc
                
                # But wait, the inner Strassen result is a 2x2 matrix.
                # The entries of this 2x2 matrix are the products P_{t,s}.
                # So C_local[lbr, lbc] gets contribution from P_{t,s}.
                # P_{t,s} is a scalar.
                # So W for the entry (global_row, global_col) should get `coeff`.
                
                # However, C_local is a 2x2 matrix.
                # The inner Strassen computes the 2x2 matrix product X*Y.
                # The entries of X*Y are the sums of the 7 products.
                # So C_local[lbr, lbc] = sum_s (coeff_s * P_{t,s}).
                # So for a fixed s, we add `coeff` to W for the entry C_local[lbr, lbc].
                
                # But C_local[lbr, lbc] is a single entry in the 4x4 matrix C?
                # No, X and Y are 2x2 matrices of 2x2 blocks.
                # So X*Y is a 2x2 matrix of 2x2 blocks.
                # Each block in X*Y is a 2x2 matrix of scalars.
                # So C_local[lbr, lbc] is a 2x2 matrix.
                # So we need to add `coeff` to W for all 4 entries in this 2x2 block.
                
                for lr in range(2):
                    for lc in range(2):
                        gr = gbr * 2 + lr
                        gc = gbc * 2 + lc
                        add_w(col, [(gr, gc, c_val)])
                        
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
