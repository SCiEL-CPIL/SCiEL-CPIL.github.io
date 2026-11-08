"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Uses Strassen-based block decomposition:
  - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
  - A*B1 uses Strassen for 4x4*4x4 (rank 49)
  - A*B2 uses Strassen for 4x4*4x4 (rank 49)
  - A*B3 uses schoolbook for 4x4*4x1 (rank 16)
  - Total rank: 49 + 49 + 16 = 114
"""

import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen-based block decomposition with rank 114.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term to the decomposition."""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen decomposition for 2x2 * 2x2 (rank 7)
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = A*B = [[c11,c12],[c21,c22]]
    #
    # m1 = (a+d)(e+h) -> c11: +1, c22: +1
    # m2 = (c+d)e     -> c21: +1, c22: -1
    # m3 = a(f-h)     -> c12: +1, c22: +1
    # m4 = d(g-e)     -> c11: +1, c21: +1
    # m5 = (a+b)h     -> c11: -1, c12: +1
    # m6 = (c-a)(e+f) -> c22: +1
    # m7 = (b-d)(g+h) -> c11: +1
    
    strassen_2x2_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def strassen_2x2_block(A_row_base, A_col_base, B_row_base, B_col_base, C_row_base, C_col_base, scale=1):
        """Add Strassen terms for a 2x2*2x2 block, scaled by 'scale'."""
        for u_coeff, v_coeff, w_coeff in strassen_2x2_terms:
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            
            for local_idx, coeff in u_coeff.items():
                li = local_idx // 2
                lj = local_idx % 2
                gi = A_row_base + li
                gj = A_col_base + lj
                u_vec[gi * m + gj] = coeff * scale
            
            for local_idx, coeff in v_coeff.items():
                lj = local_idx // 2
                lk = local_idx % 2
                gj = B_row_base + lj
                gk = B_col_base + lk
                v_vec[gj * p + gk] = coeff
            
            for local_idx, coeff in w_coeff.items():
                li = local_idx // 2
                lk = local_idx % 2
                gi = C_row_base + li
                gk = C_col_base + lk
                w_vec[gk * n + gi] = coeff
            
            add_term(u_vec, v_vec, w_vec)
    
    def strassen_4x4_block(A_row_base, A_col_base, B_row_base, B_col_base, C_row_base, C_col_base):
        """Add Strassen terms for a 4x4*4x4 block using block Strassen.
        
        Standard Strassen 4x4x4: 7 block-level multiplications, each a 2x2x2 product
        decomposed with 2x2 Strassen (7 terms each). Total: 7*7 = 49 terms.
        
        Block structure:
        A = [[A11, A12], [A21, A22]]  (each 2x2)
        B = [[B11, B12], [B21, B22]]  (each 2x2)
        C = [[C11, C12], [C21, C22]]  (each 2x2)
        
        M1 = (A11+A22)(B11+B22) -> C11: +1, C22: +1
        M2 = (A21+A22)B11       -> C21: +1, C22: -1
        M3 = A11(B12-B22)       -> C12: +1, C22: +1
        M4 = A22(B21-B11)       -> C11: +1, C21: +1
        M5 = (A11+A12)B22       -> C11: -1, C12: +1
        M6 = (A21-A11)(B11+B12) -> C22: +1
        M7 = (A12-A22)(B21+B22) -> C11: +1
        
        Each Mi is a product of two 2x2 matrices (linear combinations of blocks),
        decomposed with 2x2 Strassen.
        """
        # Block offsets in units of 2
        # A11: (0,0), A12: (0,2), A21: (2,0), A22: (2,2)
        # B11: (0,0), B12: (0,2), B21: (2,0), B22: (2,2)
        # C11: (0,0), C12: (0,2), C21: (2,0), C22: (2,2)
        
        # For each of the 7 block-level Strassen multiplications:
        # Each is of the form (sum of A blocks) * (sum of B blocks) -> sum of C blocks
        # We express each as a single 2x2x2 Strassen by treating the linear combinations
        # as the "A" and "B" of a virtual 2x2x2 product.
        
        # Actually, the correct approach: each Mi is a bilinear form
        # Mi = (linear combo of A entries) * (linear combo of B entries)
        # and contributes to specific C entries.
        # We decompose each Mi using 2x2 Strassen on the block level.
        
        # M1 = (A11+A22)(B11+B22)
        # This is a 2x2*2x2 product where:
        #   "A" = A11+A22, "B" = B11+B22
        #   Result contributes +1 to C11, +1 to C22
        # We decompose (A11+A22)(B11+B22) using 2x2 Strassen by creating
        # virtual blocks. But since A11+A22 is itself a 2x2 matrix (sum of two 2x2 blocks),
        # we can directly apply 2x2 Strassen to compute this product.
        
        # The key insight: (X+Y)(Z+W) where X,Y,Z,W are 2x2 matrices
        # = XZ + XW + YZ + YW
        # Each of these is a 2x2*2x2 product.
        # But we want to use Strassen on the combined (X+Y)*(Z+W).
        
        # Actually, the standard 4x4 Strassen works as follows:
        # We define 7 intermediate 2x2 matrices:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B12-B22)
        # P4 = A22(B21-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        
        # Each Pi is a 2x2*2x2 product, decomposed with Strassen (7 terms each).
        # Total: 7 * 7 = 49 terms.
        
        # For each Pi, we need to:
        # 1. Form the left 2x2 matrix (linear combo of A blocks)
        # 2. Form the right 2x2 matrix (linear combo of B blocks)
        # 3. Apply 2x2 Strassen to compute Pi
        # 4. Add Pi's contribution to the appropriate C blocks
        
        # Define the 7 Pi terms:
        # Each: (left_block_coeffs, right_block_coeffs, c_block_coeffs)
        # left_block_coeffs: dict mapping block (br, bc) -> coefficient
        # right_block_coeffs: dict mapping block (br, bc) -> coefficient
        # c_block_coeffs: dict mapping block (br, bc) -> coefficient
        
        pi_terms = [
            # P1 = (A11+A22)(B11+B22)
            # Left: A11: +1, A22: +1
            # Right: B11: +1, B22: +1
            # C: C11: +1, C22: +1
            ({(0,0): 1, (2,0): 1}, {(0,0): 1, (0,2): 1}, {(0,0): 1, (2,2): 1}),
            
            # P2 = (A21+A22)B11
            # Left: A21: +1, A22: +1
            # Right: B11: +1
            # C: C21: +1, C22: -1
            ({(2,0): 1, (2,0): 1}, {(0,0): 1}, {(2,0): 1, (2,2): -1}),
            
            # P3 = A11(B12-B22)
            # Left: A11: +1
            # Right: B12: +1, B22: -1
            # C: C12: +1, C22: +1
            ({(0,0): 1}, {(0,2): 1, (2,2): -1}, {(0,2): 1, (2,2): 1}),
            
            # P4 = A22(B21-B11)
            # Left: A22: +1
            # Right: B21: +1, B11: -1
            # C: C11: +1, C21: +1
            ({(2,0): 1}, {(2,0): 1, (0,0): -1}, {(0,0): 1, (2,0): 1}),
            
            # P5 = (A11+A12)B22
            # Left: A11: +1, A12: +1
            # Right: B22: +1
            # C: C11: -1, C12: +1
            ({(0,0): 1, (0,2): 1}, {(2,2): 1}, {(0,0): -1, (0,2): 1}),
            
            # P6 = (A21-A11)(B11+B12)
            # Left: A21: +1, A11: -1
            # Right: B11: +1, B12: +1
            # C: C22: +1
            ({(2,0): 1, (0,0): -1}, {(0,0): 1, (0,2): 1}, {(2,2): 1}),
            
            # P7 = (A12-A22)(B21+B22)
            # Left: A12: +1, A22: -1
            # Right: B21: +1, B22: +1
            # C: C11: +1
            ({(0,2): 1, (2,0): -1}, {(2,0): 1, (2,2): 1}, {(0,0): 1}),
        ]
        
        for left_coeffs, right_coeffs, c_coeffs in pi_terms:
            # For this Pi, we have a 2x2*2x2 product where:
            # Left 2x2 matrix = sum of left_coeffs[br,bc] * A_block(br,bc)
            # Right 2x2 matrix = sum of right_coeffs[br,bc] * B_block(br,bc)
            # Result contributes to C blocks per c_coeffs
            
            # To decompose this with 2x2 Strassen, we need to express
            # (sum of A blocks) * (sum of B blocks) as a sum of 7 rank-1 terms.
            
            # The 2x2 Strassen decomposes X*Y where X,Y are 2x2 into 7 terms.
            # Here X = sum of A blocks, Y = sum of B blocks.
            # Each A block and B block is itself a 2x2 matrix.
            
            # We can expand: X*Y = (sum_a a_a * A_a) * (sum_b b_b * B_b)
            # = sum_{a,b} a_a * b_b * (A_a * B_b)
            
            # Each A_a * B_b is a 2x2*2x2 product, decomposed with Strassen (7 terms).
            # But this would give too many terms.
            
            # Instead, we treat X and Y as single 2x2 matrices (formed by linear combination)
            # and apply 2x2 Strassen directly. The linear combinations are "free" in the
            # tensor decomposition sense.
            
            # So we create a virtual 2x2 A matrix and 2x2 B matrix, apply Strassen,
            # and scale appropriately.
            
            # For each 2x2 Strassen term (u, v, w):
            # u is coefficients for the virtual A's 4 entries
            # v is coefficients for the virtual B's 4 entries
            # w is coefficients for the result's 4 entries
            
            # The virtual A's entry (li, lj) = sum of left_coeffs[br,bc] * A_block(br,bc)[li,lj]
            # The virtual B's entry (lj, lk) = sum of right_coeffs[br,bc] * B_block(br,bc)[lj,lk]
            
            for u_coeff, v_coeff, w_coeff in strassen_2x2_terms:
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                
                # Build u_vec: for each virtual A entry (li, lj), add contributions
                # from all A blocks with their coefficients
                for local_idx, coeff in u_coeff.items():
                    li = local_idx // 2
                    lj = local_idx % 2
                    # Sum over all A blocks
                    total = 0
                    for (br, bc), a_coeff in left_coeffs.items():
                        gi = A_row_base + br + li
                        gj = A_col_base + bc + lj
                        total += a_coeff
                        # Add to u_vec
                        u_vec[gi * m + gj] += coeff * a_coeff
                
                # Build v_vec: for each virtual B entry (lj, lk), add contributions
                for local_idx, coeff in v_coeff.items():
                    lj = local_idx // 2
                    lk = local_idx % 2
                    for (br, bc), b_coeff in right_coeffs.items():
                        gj = B_row_base + br + lj
                        gk = B_col_base + bc + lk
                        v_vec[gj * p + gk] += coeff * b_coeff
                
                # Build w_vec: for each result entry (li, lk), add contributions
                for local_idx, coeff in w_coeff.items():
                    li = local_idx // 2
                    lk = local_idx % 2
                    for (br, bc), c_coeff in c_coeffs.items():
                        gi = C_row_base + br + li
                        gk = C_col_base + bc + lk
                        w_vec[gk * n + gi] += coeff * c_coeff
                
                add_term(u_vec, v_vec, w_vec)
    
    # Schoolbook for 4x4 * 4x1
    def schoolbook_4x4x1(A_row_base, A_col_base, B_row_base, B_col_base, C_row_base, C_col_base):
        """Schoolbook decomposition for 4x4 * 4x1 block."""
        for i in range(4):
            for j in range(4):
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                
                gi = A_row_base + i
                gj = A_col_base + j
                u_vec[gi * m + gj] = 1
                
                bj = B_row_base + j
                bk = B_col_base
                v_vec[bj * p + bk] = 1
                
                ci = C_row_base + i
                ck = C_col_base
                w_vec[ck * n + ci] = 1
                
                add_term(u_vec, v_vec, w_vec)
    
    # Build the full decomposition:
    # A is 4x4 (rows 0-3, cols 0-3)
    # B is 4x9 (rows 0-3, cols 0-8)
    # C is 9x4 (rows 0-8, cols 0-3)
    
    # B1 = B[:, 0:4] (4x4) -> C1 = C[:, 0:4] (4x4)
    strassen_4x4_block(
        A_row_base=0, A_col_base=0,
        B_row_base=0, B_col_base=0,
        C_row_base=0, C_col_base=0
    )
    
    # B2 = B[:, 4:8] (4x4) -> C2 = C[:, 4:8] (4x4)
    strassen_4x4_block(
        A_row_base=0, A_col_base=0,
        B_row_base=0, B_col_base=4,
        C_row_base=4, C_col_base=0
    )
    
    # B3 = B[:, 8:9] (4x1) -> C3 = C[:, 8:9] (4x1)
    schoolbook_4x4x1(
        A_row_base=0, A_col_base=0,
        B_row_base=0, B_col_base=8,
        C_row_base=8, C_col_base=0
    )
    
    # Check deadline
    if time.time() - start_time > 110:
        pass
    
    # Convert to row-major format: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    rank = len(U_cols)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for i in range(n * m):
            U[i][c] = U_cols[c][i]
        for j in range(m * p):
            V[j][c] = V_cols[c][j]
        for k in range(p * n):
            W[k][c] = W_cols[c][k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
