import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 100  # safety margin
    
    def build_strassen_block(i0, j0, k0, U, V, W, col_start):
        """
        Add a Strassen 2x2x2 decomposition for block at offsets (i0, j0, k0).
        Local indices: i in {0,1}, j in {0,1}, k in {0,1}
        Global:
          U row: (i0+i)*m + (j0+j)  where m=4
          V row: (j0+j)*p + (k0+k)  where p=4
          W row: (k0+k)*n + (i0+i)  where n=2
        """
        n, m, p = DIMENSIONS
        
        def u_idx(i, j):
            return (i0 + i) * m + (j0 + j)
        
        def v_idx(j, k):
            return (j0 + j) * p + (k0 + k)
        
        def w_idx(k, i):
            return (k0 + k) * n + (i0 + i)
        
        # Strassen 7 multiplications for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 - M2 + M4 + M7
        # C01 = M3 + M5
        # C10 = M6 + M2
        # C11 = M1 - M3 + M5 - M4
        
        # Each M_r contributes to U, V, W with coefficients
        # U[row, col] += coeff for A
        # V[row, col] += coeff for B
        # W[row, col] += coeff for C
        
        # We'll define the contributions for each of the 7 multiplications
        # For each multiplication r, we have:
        #   U contribution: list of (row, coeff)
        #   V contribution: list of (row, coeff)
        #   W contribution: list of (row, coeff)
        
        strassen_defs = [
            # M1: (A00+A11)(B00+B11)
            {
                'U': [(u_idx(0,0), 1), (u_idx(1,1), 1)],
                'V': [(v_idx(0,0), 1), (v_idx(1,1), 1)],
                'W': [(w_idx(0,0), 1), (w_idx(1,1), 1)]  # C00 += M1, C11 += M1
            },
            # M2: (A10+A11)B00
            {
                'U': [(u_idx(1,0), 1), (u_idx(1,1), 1)],
                'V': [(v_idx(0,0), 1)],
                'W': [(w_idx(0,0), -1), (w_idx(1,0), 1)]  # C00 -= M2, C10 += M2
            },
            # M3: A00(B10-B11)
            {
                'U': [(u_idx(0,0), 1)],
                'V': [(v_idx(1,0), 1), (v_idx(1,1), -1)],
                'W': [(w_idx(0,1), 1), (w_idx(1,1), -1)]  # C01 += M3, C11 -= M3
            },
            # M4: A11(B01-B10)
            {
                'U': [(u_idx(1,1), 1)],
                'V': [(v_idx(0,1), 1), (v_idx(0,0), -1)],
                'W': [(w_idx(0,0), 1), (w_idx(1,1), -1)]  # C00 += M4, C11 -= M4
            },
            # M5: (A00+A01)B11
            {
                'U': [(u_idx(0,0), 1), (u_idx(0,1), 1)],
                'V': [(v_idx(1,1), 1)],
                'W': [(w_idx(0,1), 1), (w_idx(1,1), 1)]  # C01 += M5, C11 += M5
            },
            # M6: (A10-A00)(B00+B10)
            {
                'U': [(u_idx(1,0), 1), (u_idx(0,0), -1)],
                'V': [(v_idx(0,0), 1), (v_idx(1,0), 1)],
                'W': [(w_idx(1,0), 1)]  # C10 += M6
            },
            # M7: (A01-A11)(B01+B11)
            {
                'U': [(u_idx(0,1), 1), (u_idx(1,1), -1)],
                'V': [(v_idx(0,1), 1), (v_idx(1,1), 1)],
                'W': [(w_idx(0,0), 1)]  # C00 += M7
            },
        ]
        
        for r, defn in enumerate(strassen_defs):
            col = col_start + r
            for row, coeff in defn['U']:
                U[row][col] += coeff
            for row, coeff in defn['V']:
                V[row][col] += coeff
            for row, coeff in defn['W']:
                W[row][col] += coeff
        
        return col_start + 7
    
    n, m, p = DIMENSIONS
    
    # Use Strassen on four 2x2x2 blocks
    # Block structure: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # Split j into {0,1} and {2,3}, k into {0,1} and {2,3}
    # So 4 blocks: (i0=0, j0=0, k0=0), (i0=0, j0=0, k0=2), (i0=0, j0=2, k0=0), (i0=0, j0=2, k0=2)
    # Wait, i0 is always 0 since i only has 2 values
    # Actually, let me reconsider the block structure.
    
    # The matrix A is 2x4, B is 4x4, C is 4x2.
    # We can view this as:
    # A = [A_0 | A_1] where A_0, A_1 are 2x2
    # B = [B_00 B_01; B_10 B_11] where each B_ij is 2x2
    # C = A @ B = [A_0 B_00 + A_1 B_10, A_0 B_01 + A_1 B_11]
    # So C is 2x4, split into two 2x2 blocks: C_0 and C_1
    
    # C_0 = A_0 B_00 + A_1 B_10 (2x2)
    # C_1 = A_0 B_01 + A_1 B_11 (2x2)
    
    # We can apply Strassen to each 2x2 output block separately.
    # For C_0: it's a sum of two products of 2x2 matrices.
    # We can use the standard rank-7 decomposition for 2x2x2, but we have two terms.
    # Actually, the product of two 2x2 matrices has rank at most 7.
    # The sum of two such products doesn't directly give rank 7.
    
    # Let me think differently. We have 4 independent 2x2x2 matrix multiplications:
    # 1. A_0 @ B_00 contributes to C_0
    # 2. A_1 @ B_10 contributes to C_0  
    # 3. A_0 @ B_01 contributes to C_1
    # 4. A_1 @ B_11 contributes to C_1
    
    # Each requires 7 multiplications if done separately, total 28.
    # But we can potentially share some computations.
    
    # For now, let's just use 4 separate Strassen decompositions, total rank 28.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 2*4 = 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 4*4 = 16 rows
    W = [[0] * rank for _ in range(p * n)]  # 4*2 = 8 rows
    
    col = 0
    # Block 1: A_0 @ B_00 -> contributes to C_0
    col = build_strassen_block(0, 0, 0, U, V, W, col)  # i0=0, j0=0, k0=0
    # Block 2: A_1 @ B_10 -> contributes to C_0
    col = build_strassen_block(0, 2, 0, U, V, W, col)  # i0=0 (but A_1 means i=1 locally, so this is handled by the U indexing), j0=2, k0=0
    # Wait, I need to be more careful. A_1 is the second 2x2 block of A, which corresponds to rows i=1 in local coords when i0=0.
    # Actually, my build_strassen_block assumes the local 2x2 block starts at (i0, j0, k0).
    # For A_0 @ B_00: A_0 is rows 0,1 of A (local i=0,1 with i0=0), B_00 is cols 0,1 of B (local j=0,1 with j0=0), output to C_0 which is rows 0,1 of C (local k=0,1 with k0=0).
    # For A_1 @ B_10: A_1 is also rows 0,1 of A (since A is 2x4, A_0 and A_1 are both 2x2, so A_1 uses the same i indices 0,1 but different j). 
    # Hmm, this is getting confusing. Let me redefine.
    
    # A is 2x4. Split A = [A_left | A_right] where A_left = A[:, 0:2], A_right = A[:, 2:4].
    # B is 4x4. Split B into 2x2 blocks: B_00 = B[0:2, 0:2], B_01 = B[0:2, 2:4], B_10 = B[2:4, 0:2], B_11 = B[2:4, 2:4].
    # C = A @ B = [A_left @ B_00 + A_right @ B_10 | A_left @ B_01 + A_right @ B_11].
    # C is 2x4. C_left = C[:, 0:2], C_right = C[:, 2:4].
    
    # So:
    # C_left = A_left @ B_00 + A_right @ B_10
    # C_right = A_left @ B_01 + A_right @ B_11
    
    # Each of these is a 2x2 matrix that is the sum of two 2x2 matrix products.
    
    # For C_left: we have two terms: A_left @ B_00 and A_right @ B_10.
    # We can use Strassen for each product separately, giving 7+7=14 for C_left, and similarly 14 for C_right, total 28.
    
    # Let me redo with correct block definitions.
    # U row: (i, j) where i in 0..1, j in 0..3
    # V row: (j, k) where j in 0..3, k in 0..3
    # W row: (k, i) where k in 0..3, i in 0..1
    
    # Block 1: A_left @ B_00
    # A_left: i in {0,1}, j in {0,1}
    # B_00: j in {0,1}, k in {0,1}
    # Output: k in {0,1}, i in {0,1}
    col = 0
    col = build_strassen_block(0, 0, 0, U, V, W, col)
    
    # Block 2: A_right @ B_10
    # A_right: i in {0,1}, j in {2,3}
    # B_10: j in {2,3}, k in {0,1}
    # Output: k in {0,1}, i in {0,1}
    col = build_strassen_block(0, 2, 0, U, V, W, col)
    
    # Block 3: A_left @ B_01
    # A_left: i in {0,1}, j in {0,1}
    # B_01: j in {0,1}, k in {2,3}
    # Output: k in {2,3}, i in {0,1}
    col = build_strassen_block(0, 0, 2, U, V, W, col)
    
    # Block 4: A_right @ B_11
    # A_right: i in {0,1}, j in {2,3}
    # B_11: j in {2,3}, k in {2,3}
    # Output: k in {2,3}, i in {0,1}
    col = build_strassen_block(0, 2, 2, U, V, W, col)
    
    # Verify: col should be 28
    assert col == 28, f"Expected 28 columns, got {col}"
    
    # Convert to lists (already are lists)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
