# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for a rank-94 decomposition of 5x5x5 matrix multiplication.
    Uses Strassen's algorithm for the 4x4 core (rank 49) and schoolbook for the cross terms.
    """
    n, m, p = DIMENSIONS
    # Total rank: 49 (Strassen 4x4) + 20 (C[:,4]) + 20 (C[4,:]) + 5 (C[4,4]) = 94
    rank = 94
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: 4x4 Strassen Core ---
    # Indices i,j,k in 0..3
    
    # Define the 7 Strassen terms for 2x2
    # Returns list of (coeff_A, coeff_B, coeff_C) for the 4 output blocks
    # Structure: term_idx -> (L_coeffs, R_coeffs, C_coeffs)
    # L_coeffs: 2x2 matrix of coefficients for A blocks (00, 01, 10, 11)
    # R_coeffs: 2x2 matrix of coefficients for B blocks
    # C_coeffs: 2x2 matrix of coefficients for C blocks
    
    # We will generate the 49 terms by combining outer and inner Strassen.
    
    # Helper to get Strassen coefficients for a single 2x2 multiplication
    def get_strassen_coeffs():
        # Returns list of 7 tuples: (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs: dict mapping (i,j) to coeff for A_ij
        # V_coeffs: dict mapping (j,k) to coeff for B_jk
        # W_coeffs: dict mapping (k,i) to coeff for C_ki (note: W index is k*n+i)
        # Actually, let's just return the linear combinations for the 7 products M1..M7
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to expand these into the full 4x4 context later.
        # For now, we just need the structure.
        pass

    # Instead of abstract, we iterate over the 7 outer terms and 7 inner terms.
    
    # Outer Strassen terms (operating on 2x2 blocks)
    # Let A = [[A00, A01], [A10, A11]] where each Ai is 2x2.
    # B = [[B00, B01], [B10, B11]]
    
    # Outer M1 = (A00+A11)(B00+B11)
    # Outer M2 = (A10+A11)B00
    # Outer M3 = A00(B01-B11)
    # Outer M4 = A11(B10-B00)
    # Outer M5 = (A00+A01)B11
    # Outer M6 = (A10-A00)(B00+B01)
    # Outer M7 = (A01-A11)(B10+B11)
    
    # For each outer term t, we have:
    # X_t = sum c_i A_i  (2x2 matrix)
    # Y_t = sum d_j B_j  (2x2 matrix)
    # We compute Z_t = X_t Y_t using Inner Strassen.
    # Inner Strassen on X_t, Y_t produces 7 products.
    # Each product is a scalar linear combo of X_t entries * Y_t entries.
    # Each X_t entry is a linear combo of A elements.
    # So each product is a linear combo of A elements * linear combo of B elements.
    
    # We will store the coefficients for each of the 49 terms.
    
    # Define the outer term definitions
    # format: (L_blocks, R_blocks) where L_blocks is dict {block_idx: coeff}
    # block_idx: 0->00, 1->01, 2->10, 3->11
    outer_terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    # Define inner Strassen terms for 2x2
    # These are the same as the outer ones, but applied to scalars.
    # We need to map the 2x2 indices (0,1) to the global 4x4 indices.
    # Block 0 (00): i in {0,1}, j in {0,1}
    # Block 1 (01): i in {0,1}, j in {2,3}
    # Block 2 (10): i in {2,3}, j in {0,1}
    # Block 3 (11): i in {2,3}, j in {2,3}
    
    # For a given outer term t, X_t is a 2x2 matrix.
    # X_t[0,0] = sum c_b A_b[0,0]
    # X_t[0,1] = sum c_b A_b[0,1]
    # X_t[1,0] = sum c_b A_b[1,0]
    # X_t[1,1] = sum c_b A_b[1,1]
    
    # Similarly for Y_t.
    
    # We iterate t in 0..6, u in 0..6.
    # For each pair, we generate 1 rank-1 term.
    
    # Inner Strassen formulas for X, Y -> Z
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # We need the coefficients for M1..M7 in terms of X and Y entries.
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    # We will build U, V, W contributions for each (t,u).
    
    for t in range(7):
        L_blocks, R_blocks = outer_terms[t]
        
        # Construct X and Y as 2x2 matrices of "linear forms"
        # X[i][j] is a dict mapping (global_i, global_j) to coeff
        X = [[{} for _ in range(2)] for _ in range(2)]
        Y = [[{} for _ in range(2)] for _ in range(2)]
        
        # Fill X
        for b, coeff in L_blocks.items():
            # Block b corresponds to 2x2 submatrix
            # b=0: i0,j0; i0,j1; i1,j0; i1,j1 (relative to block start)
            # Global indices:
            # b=0: i in 0,1; j in 0,1
            # b=1: i in 0,1; j in 2,3
            # b=2: i in 2,3; j in 0,1
            # b=3: i in 2,3; j in 2,3
            
            bi, bj = divmod(b, 2) # 0,0 / 0,1 / 1,0 / 1,1
            base_i = bi * 2
            base_j = bj * 2
            
            for di in range(2):
                for dj in range(2):
                    gi = base_i + di
                    gj = base_j + dj
                    # X[di][dj] += coeff * A[gi][gj]
                    if (gi, gj) not in X[di][dj]:
                        X[di][dj][(gi, gj)] = 0
                    X[di][dj][(gi, gj)] += coeff
        
        # Fill Y
        for b, coeff in R_blocks.items():
            bi, bj = divmod(b, 2)
            base_i = bi * 2
            base_j = bj * 2
            for di in range(2):
                for dj in range(2):
                    gi = base_i + di
                    gj = base_j + dj
                    # Y[di][dj] += coeff * B[gi][gj]
                    if (gi, gj) not in Y[di][dj]:
                        Y[di][dj][(gi, gj)] = 0
                    Y[di][dj][(gi, gj)] += coeff
        
        # Now apply Inner Strassen to X, Y
        # We need to generate 7 terms (u=0..6)
        # Each term u corresponds to a product M_u
        # We define the linear combinations for X and Y for each M_u
        
        # M1: (X00+X11) (Y00+Y11)
        # M2: (X10+X11) Y00
        # M3: X00 (Y01-Y11)
        # M4: X11 (Y10-Y00)
        # M5: (X00+X01) Y11
        # M6: (X10-X00) (Y00+Y01)
        # M7: (X01-X11) (Y10+Y11)
        
        # We will create a list of (X_combo, Y_combo) for the 7 inner terms
        # X_combo: dict (gi,gj) -> coeff
        # Y_combo: dict (gi,gj) -> coeff
        
        inner_defs = [
            # (X_indices, X_coeffs, Y_indices, Y_coeffs)
            # M1
            ([ (0,0), (1,1) ], [1, 1], [ (0,0), (1,1) ], [1, 1]),
            # M2
            ([ (1,0), (1,1) ], [1, 1], [ (0,0) ], [1]),
            # M3
            ([ (0,0) ], [1], [ (0,1), (1,1) ], [1, -1]),
            # M4
            ([ (1,1) ], [1], [ (1,0), (0,0) ], [1, -1]),
            # M5
            ([ (0,0), (0,1) ], [1, 1], [ (1,1) ], [1]),
            # M6
            ([ (1,0), (0,0) ], [1, -1], [ (0,0), (0,1) ], [1, 1]),
            # M7
            ([ (0,1), (1,1) ], [1, -1], [ (1,0), (1,1) ], [1, 1]),
        ]
        
        for u in range(7):
            x_idxs, x_coeffs, y_idxs, y_coeffs = inner_defs[u]
            
            # Combine X linear forms
            U_combo = {}
            for idx, c in zip(x_idxs, x_coeffs):
                # X[idx] is a dict
                for (gi, gj), val in X[idx[0]][idx[1]].items():
                    if (gi, gj) not in U_combo:
                        U_combo[(gi, gj)] = 0
                    U_combo[(gi, gj)] += c * val
            
            # Combine Y linear forms
            V_combo = {}
            for idx, c in zip(y_idxs, y_coeffs):
                for (gi, gj), val in Y[idx[0]][idx[1]].items():
                    if (gi, gj) not in V_combo:
                        V_combo[(gi, gj)] = 0
                    V_combo[(gi, gj)] += c * val
            
            # Now we have a rank-1 term:
            # sum_{i,j} U_combo[i,j] A[i,j] * sum_{j,k} V_combo[j,k] B[j,k]
            # This contributes to C.
            # The contribution to C[i,k] is sum_j U_combo[i,j] V_combo[j,k]
            
            # We add this to U, V, W at column 'col'
            for (i, j), coeff in U_combo.items():
                U[i * m + j][col] += coeff
            for (j, k), coeff in V_combo.items():
                V[j * p + k][col] += coeff
            
            # Compute W coefficients
            # W[k*n + i] += sum_j U_combo[i,j] V_combo[j,k]
            # We iterate over all i, k
            for i in range(4):
                for k in range(4):
                    w_val = 0
                    for j in range(4):
                        u_val = U_combo.get((i, j), 0)
                        v_val = V_combo.get((j, k), 0)
                        w_val += u_val * v_val
                    if w_val != 0:
                        W[k * n + i][col] += w_val
            
            col += 1
    
    # --- Part 2: Cross Terms ---
    # C[:,4] = A[:,:4] B[4,4] + A[:,4] B[4,4] ? No.
    # C[i,4] = sum_{j=0}^3 A[i,j] B[j,4] + A[i,4] B[4,4]
    
    # Terms for sum_{j=0}^3 A[i,j] B[j,4]
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    # Terms for A[i,4] B[4,4]
    for i in range(4):
        U[i * m + 4][col] = 1
        V[4 * p + 4][col] = 1
        W[4 * n + i][col] = 1
        col += 1
        
    # C[4,j] = sum_{k=0}^3 A[4,k] B[k,j] + A[4,4] B[4,j]
    
    # Terms for sum_{k=0}^3 A[4,k] B[k,j]
    for j in range(4):
        for k in range(4):
            U[4 * m + k][col] = 1
            V[k * p + j][col] = 1
            W[j * n + 4][col] = 1
            col += 1
            
    # Terms for A[4,4] B[4,j]
    for j in range(4):
        U[4 * m + 4][col] = 1
        V[4 * p + j][col] = 1
        W[j * n + 4][col] = 1
        col += 1
        
    # C[4,4] = sum_{k=0}^3 A[4,k] B[k,4] + A[4,4] B[4,4]
    
    # Terms for sum_{k=0}^3 A[4,k] B[k,4]
    for k in range(4):
        U[4 * m + k][col] = 1
        V[k * p + 4][col] = 1
        W[4 * n + 4][col] = 1
        col += 1
        
    # Term for A[4,4] B[4,4]
    U[4 * m + 4][col] = 1
    V[4 * p + 4][col] = 1
    W[4 * n + 4][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
