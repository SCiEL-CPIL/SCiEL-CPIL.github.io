import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We use a hybrid approach:
    # Split B into B1 (4x4) and B2 (4x3).
    # C = A B1 + A B2.
    # A B1 uses Strassen (Rank 49).
    # A B2 uses Schoolbook (Rank 48).
    # Total Rank 97.
    
    rank_strassen = 49
    rank_schoolbook = 48
    total_rank = rank_strassen + rank_schoolbook
    
    U = [[0] * total_rank for _ in range(n * m)] # 16 x 97
    V = [[0] * total_rank for _ in range(m * p)] # 28 x 97
    W = [[0] * total_rank for _ in range(p * n)] # 28 x 97
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w, col):
        U[u_idx][col] = coeff_u
        V[v_idx][col] = coeff_v
        W[w_idx][col] = coeff_w

    # --- Part 1: Strassen for A (4x4) * B1 (4x4) ---
    # B1 corresponds to B columns 0..3.
    # Strassen on 2x2 blocks.
    # A blocks:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # B1 blocks:
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B10: rows 2-3, cols 0-1
    # B11: rows 2-3, cols 2-3
    
    # Strassen terms M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C blocks:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to the global U, V, W indices.
    # U index: i*m + j. i in 0..3, j in 0..3.
    # V index: j*p + k. j in 0..3, k in 0..6.
    # W index: k*n + i. k in 0..6, i in 0..3.
    
    # Strassen operates on 2x2 sub-matrices.
    # We will generate the rank-7 terms for each of the 7 Strassen multiplications.
    # Each Strassen multiplication is a 2x2x2 tensor.
    # The 2x2x2 tensor has rank 7.
    # We need to embed these 7 terms into the 4x4x4 space.
    
    # Let's define the 7 basis vectors for the 2x2x2 Strassen algorithm.
    # For a 2x2x2 multiplication X*Y=Z:
    # x00, x01, x10, x11
    # y00, y01, y10, y11
    # z00, z01, z10, z11
    
    # The 7 terms (u, v, w) where u is 4-dim (X coeffs), v is 4-dim (Y coeffs), w is 4-dim (Z coeffs).
    # Actually, the tensor is sum of u_r \otimes v_r \otimes w_r.
    # u_r is a linear form in X. v_r in Y. w_r is a linear form in Z?
    # No, w_r is the output matrix contribution.
    # In the decomposition C = sum (L_r(A) M_r(B)) W_r, W_r is a matrix.
    # So for each term, we add L_r(A) to U, M_r(B) to V, and the matrix W_r to W.
    
    # Strassen terms for 2x2x2:
    # 1. L = x00+x11, M = y00+y11, W = [[1, -1], [-1, 1]]? No.
    # Let's use the standard Strassen formulas for the products.
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y01-y11)
    # M4 = x11(y10-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to construct the U, V, W entries for these 7 terms.
    # For each term r, we have:
    # U_part: vector of coeffs for X.
    # V_part: vector of coeffs for Y.
    # W_part: 2x2 matrix of coeffs for Z.
    
    # Term 1:
    # U: [1, 0, 0, 1] (x00, x01, x10, x11) -> x00+x11
    # V: [1, 0, 0, 1] (y00, y01, y10, y11) -> y00+y11
    # W: [[1, 0], [0, 1]]? No, M1 contributes to C00 (+1) and C11 (+1).
    # So W = [[1, 0], [0, 1]]?
    # C00 gets +M1. C11 gets +M1.
    # Yes.
    
    # Term 2:
    # U: [0, 0, 1, 1] -> x10+x11
    # V: [1, 0, 0, 0] -> y00
    # W: [[0, 0], [1, 0]]? M2 contributes to C10 (+1).
    # C10 is row 1, col 0 of the 2x2 block.
    # So W = [[0, 0], [1, 0]].
    
    # Term 3:
    # U: [1, 0, 0, 0] -> x00
    # V: [0, 1, 0, -1] -> y01-y11
    # W: [[0, 1], [0, 0]]? M3 contributes to C01 (+1).
    # C01 is row 0, col 1.
    # So W = [[0, 1], [0, 0]].
    
    # Term 4:
    # U: [0, 0, 0, 1] -> x11
    # V: [0, 0, 1, -1] -> y10-y00
    # W: [[1, 0], [1, 0]]? M4 contributes to C00 (+1) and C10 (+1).
    # So W = [[1, 0], [1, 0]].
    
    # Term 5:
    # U: [1, 1, 0, 0] -> x00+x01
    # V: [0, 0, 0, 1] -> y11
    # W: [[-1, 0], [0, 1]]? M5 contributes to C00 (-1) and C01 (+1).
    # So W = [[-1, 0], [0, 1]].
    
    # Term 6:
    # U: [0, 0, 1, -1] -> x10-x00
    # V: [1, 1, 0, 0] -> y00+y01
    # W: [[0, 0], [0, 1]]? M6 contributes to C11 (+1).
    # So W = [[0, 0], [0, 1]].
    
    # Term 7:
    # U: [0, 1, 0, -1] -> x01-x11
    # V: [0, 0, 1, 1] -> y10+y11
    # W: [[1, 0], [0, 0]]? M7 contributes to C00 (+1).
    # So W = [[1, 0], [0, 0]].

    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [[1, 0], [0, 1]]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [[0, 0], [1, 0]]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [[0, 1], [0, 0]]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [[1, 0], [1, 0]]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [[-1, 0], [0, 1]]),
        ([0, 0, 1, -1], [1, 1, 0, 0], [[0, 0], [0, 1]]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [[1, 0], [0, 0]])
    ]
    
    # Now we map these to the 4x4 blocks.
    # Block (br, bc) of A corresponds to rows br*2..br*2+1, cols bc*2..bc*2+1.
    # Block (br, bc) of B1 corresponds to rows br*2..br*2+1, cols bc*2..bc*2+1.
    # Block (br, bc) of C1 corresponds to rows br*2..br*2+1, cols bc*2..bc*2+1.
    
    # Strassen combines blocks:
    # M1 uses A00, A11 and B00, B11.
    # A00 is block (0,0). A11 is block (1,1).
    # B00 is block (0,0). B11 is block (1,1).
    # The result M1 is added to C00 (block 0,0) and C11 (block 1,1).
    
    # We need to iterate through the 7 Strassen terms and the 7 block combinations?
    # No, Strassen is a single algorithm for the 4x4 multiplication.
    # The 7 terms M1..M7 are computed from the 4x4 blocks.
    # Each M_r is a 2x2 matrix product? No.
    # M1 = (A00+A11)(B00+B11). This is a 2x2 matrix product.
    # So we need to decompose M1 into 7 rank-1 terms (Strassen for 2x2x2).
    # Then we map those rank-1 terms back to the 4x4 indices.
    
    # So, for each of the 7 "meta-terms" (M1..M7), we generate 7 "sub-terms" (Strassen for 2x2x2).
    # Total 49 terms.
    
    # Meta-term 1: X = A00+A11, Y = B00+B11.
    # X is a 2x2 matrix formed by summing two 2x2 blocks.
    # Y is a 2x2 matrix formed by summing two 2x2 blocks.
    # We apply the 7 Strassen sub-terms to X and Y.
    # For each sub-term s (with U_s, V_s, W_s):
    # U_global = U_s applied to X.
    # V_global = V_s applied to Y.
    # W_global = W_s applied to the output blocks C00 and C11.
    
    # Let's define the block mappings.
    # A blocks:
    # A00: i in 0,1; j in 0,1
    # A01: i in 0,1; j in 2,3
    # A10: i in 2,3; j in 0,1
    # A11: i in 2,3; j in 2,3
    
    # B1 blocks:
    # B00: j in 0,1; k in 0,1
    # B01: j in 0,1; k in 2,3
    # B10: j in 2,3; k in 0,1
    # B11: j in 2,3; k in 2,3
    
    # C1 blocks:
    # C00: i in 0,1; k in 0,1
    # C01: i in 0,1; k in 2,3
    # C10: i in 2,3; k in 0,1
    # C11: i in 2,3; k in 2,3
    
    # Meta-terms definitions (X_blocks, Y_blocks, W_blocks):
    # M1: X=A00+A11, Y=B00+B11, W=C00+C11
    # M2: X=A10+A11, Y=B00, W=C10
    # M3: X=A00, Y=B01-B11, W=C01
    # M4: X=A11, Y=B10-B00, W=C00+C10
    # M5: X=A00+A01, Y=B11, W=-C00+C01
    # M6: X=A10-A00, Y=B00+B01, W=C11
    # M7: X=A01-A11, Y=B10+B11, W=C00
    
    meta_terms = [
        # (X_blocks, Y_blocks, W_blocks)
        # X_blocks: list of (block_idx, coeff)
        # Y_blocks: list of (block_idx, coeff)
        # W_blocks: list of (block_idx, coeff)
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]), # M1
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1) ]), # M2
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1) ]), # M3
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, 1) ]), # M4
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (0, -1), (1, 1) ]), # M5
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (3, 1) ]), # M6
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1) ]) # M7
    ]
    
    # Block indices: 0=A00/B00/C00, 1=A01/B01/C01, 2=A10/B10/C10, 3=A11/B11/C11
    # Block (br, bc) maps to:
    # A: rows br*2..br*2+1, cols bc*2..bc*2+1
    # B: rows br*2..br*2+1, cols bc*2..bc*2+1
    # C: rows br*2..br*2+1, cols bc*2..bc*2+1
    
    col = 0
    for meta in meta_terms:
        x_blocks, y_blocks, w_blocks = meta
        # For each of the 7 Strassen sub-terms
        for u_s, v_s, w_s in strassen_terms:
            # u_s is [x00, x01, x10, x11]
            # v_s is [y00, y01, y10, y11]
            # w_s is [[z00, z01], [z10, z11]]
            
            # Construct U, V, W vectors for this term
            # U has 16 elements (4x4 A)
            # V has 28 elements (4x7 B)
            # W has 28 elements (4x4 C1 part, but we only use k=0..3 here)
            
            # Initialize local vectors
            u_vec = [0] * 16
            v_vec = [0] * 28
            w_vec = [0] * 28 # Only first 16 will be used for C1
            
            # Fill U from x_blocks
            for b_idx, coeff in x_blocks:
                br = b_idx // 2
                bc = b_idx % 2
                # x_s[0]=x00, x_s[1]=x01, x_s[2]=x10, x_s[3]=x11
                # x00 corresponds to local row 0, col 0 of the block
                # Global row br*2 + r, col bc*2 + c
                u_vec[(br*2 + 0)*4 + (bc*2 + 0)] += u_s[0] * coeff
                u_vec[(br*2 + 0)*4 + (bc*2 + 1)] += u_s[1] * coeff
                u_vec[(br*2 + 1)*4 + (bc*2 + 0)] += u_s[2] * coeff
                u_vec[(br*2 + 1)*4 + (bc*2 + 1)] += u_s[3] * coeff
                
            # Fill V from y_blocks
            for b_idx, coeff in y_blocks:
                br = b_idx // 2
                bc = b_idx % 2
                # y00 corresponds to local row 0, col 0
                # Global row br*2 + r, col bc*2 + c
                v_vec[(br*2 + 0)*7 + (bc*2 + 0)] += v_s[0] * coeff
                v_vec[(br*2 + 0)*7 + (bc*2 + 1)] += v_s[1] * coeff
                v_vec[(br*2 + 1)*7 + (bc*2 + 0)] += v_s[2] * coeff
                v_vec[(br*2 + 1)*7 + (bc*2 + 1)] += v_s[3] * coeff
                
            # Fill W from w_blocks
            for b_idx, coeff in w_blocks:
                br = b_idx // 2
                bc = b_idx % 2
                # w_s[0][0]=z00, w_s[0][1]=z01, w_s[1][0]=z10, w_s[1][1]=z11
                # Global row br*2 + r, col bc*2 + c
                # W index: k*n + i. k is col, i is row.
                # C is 4x4. k in 0..3, i in 0..3.
                w_vec[(bc*2 + 0)*4 + (br*2 + 0)] += w_s[0][0] * coeff
                w_vec[(bc*2 + 1)*4 + (br*2 + 0)] += w_s[0][1] * coeff
                w_vec[(bc*2 + 0)*4 + (br*2 + 1)] += w_s[1][0] * coeff
                w_vec[(bc*2 + 1)*4 + (br*2 + 1)] += w_s[1][1] * coeff
                
            # Add to global U, V, W
            for i in range(16):
                U[i][col] = u_vec[i]
            for j in range(28):
                V[j][col] = v_vec[j]
            for k in range(28):
                W[k][col] = w_vec[k]
                
            col += 1
            
    # --- Part 2: Schoolbook for A (4x4) * B2 (4x3) ---
    # B2 corresponds to B columns 4..6.
    # A is 4x4.
    # C2 is 4x3.
    # Rank 48.
    
    for i in range(4):
        for j in range(4):
            for k in range(3):
                # A[i,j] * B[j, k+4]
                u_idx = i * 4 + j
                v_idx = j * 7 + (k + 4)
                w_idx = (k + 4) * 4 + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
