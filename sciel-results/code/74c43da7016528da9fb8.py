import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Define the 7 Strassen steps
    # Each step is (X_coeffs, Y_coeffs, C_targets)
    # X_coeffs: dict mapping (i, j) in A to coeff
    # Y_coeffs: dict mapping (j, k) in B to coeff
    # C_targets: list of (block_idx, sign) where block_idx 0=C11, 1=C12, 2=C21, 3=C22
    
    steps = [
        # Step 1: X = A11+A22, Y = B11+B22, C = C11+C22
        (
            {(0,0): 1, (0,1): 1, (1,2): 1, (1,3): 1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, 1), (3, 1)]
        ),
        # Step 2: X = A21+A22, Y = B11, C = C21-C22
        (
            {(1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1},
            [(2, 1), (3, -1)]
        ),
        # Step 3: X = A11, Y = B12-B22, C = C12+C22
        (
            {(0,0): 1, (0,1): 1},
            {(0,3): 1, (0,4): 1, (0,5): 1, (1,3): 1, (1,4): 1, (1,5): 1, (2,3): -1, (2,4): -1, (2,5): -1, (3,3): -1, (3,4): -1, (3,5): -1},
            [(1, 1), (3, 1)]
        ),
        # Step 4: X = A22, Y = B21-B11, C = C11+C21
        (
            {(1,2): 1, (1,3): 1},
            {(2,0): 1, (2,1): 1, (2,2): 1, (3,0): 1, (3,1): 1, (3,2): 1, (0,0): -1, (0,1): -1, (0,2): -1, (1,0): -1, (1,1): -1, (1,2): -1},
            [(0, 1), (2, 1)]
        ),
        # Step 5: X = A11+A12, Y = B22, C = -C11+C12
        (
            {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1},
            {(2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, -1), (1, 1)]
        ),
        # Step 6: X = A21-A11, Y = B11+B12, C = C22
        (
            {(1,0): 1, (1,1): 1, (0,0): -1, (0,1): -1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1, (0,4): 1, (0,5): 1, (1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1, (1,4): 1, (1,5): 1},
            [(3, 1)]
        ),
        # Step 7: X = A12-A22, Y = B21+B22, C = C11
        (
            {(0,2): 1, (0,3): 1, (1,2): -1, (1,3): -1},
            {(2,0): 1, (2,1): 1, (2,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,0): 1, (3,1): 1, (3,2): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, 1)]
        ),
    ]
    
    col = 0
    for X_dict, Y_dict, C_targets in steps:
        # Extract x0, x1 from X_dict
        # X is 1x2. x0 is col 0, x1 is col 1.
        # But X_dict maps (i,j) to coeff.
        # We need to form the vector x0 and x1 as linear combinations of A entries.
        # x0 = sum coeff * A[i,j] for j=0? No.
        # The block A11 is [A00, A01]. A22 is [A12, A13].
        # X1 = A11+A22 = [A00+A12, A01+A13].
        # So x0 = A00+A12, x1 = A01+A13.
        
        # We need to identify which (i,j) belong to x0 and which to x1.
        # In the block structure, the "column" of the block is the second index of the block.
        # Block A11 has cols 0,1. Block A12 has cols 2,3.
        # So for X, the first component (index 0 of the 1x2 vector) corresponds to the first column of the blocks involved.
        # The second component (index 1) corresponds to the second column.
        
        # Let's map X_dict entries to x0 and x1.
        x0_coeffs = {}
        x1_coeffs = {}
        for (i, j), c in X_dict.items():
            # j is 0,1,2,3.
            # If j is 0 or 2, it's the first column of its block (A11 or A12 or A21 or A22).
            # If j is 1 or 3, it's the second column.
            if j % 2 == 0:
                x0_coeffs[(i, j)] = c
            else:
                x1_coeffs[(i, j)] = c
                
        # Similarly for Y.
        # Y is 2x3.
        # y00, y01, y02 are row 0.
        # y10, y11, y12 are row 1.
        # The blocks B11, B12, B21, B22 are 2x3.
        # B11 rows 0,1. B21 rows 2,3.
        # So row 0 of Y corresponds to rows 0,2 of B?
        # No, Y is a linear combination of 2x3 blocks.
        # So Y[0,0] is sum of coeffs for B[r,0] where r is in {0,2} depending on the block.
        # Actually, the blocks are 2x3.
        # B11 = B[0:2, 0:3].
        # B21 = B[2:4, 0:3].
        # So Y[0,0] gets contributions from B[0,0] (from B11) and B[2,0] (from B21).
        # Y[1,0] gets contributions from B[1,0] (from B11) and B[3,0] (from B21).
        
        # So we can just iterate over Y_dict and assign to y00..y12 based on (j,k).
        # y_row = j % 2? No.
        # B11 is rows 0,1. B21 is rows 2,3.
        # So if j is 0 or 2, it's the first row of the block.
        # If j is 1 or 3, it's the second row of the block.
        # And k is 0,1,2 within the block?
        # B11 cols 0,1,2. B12 cols 3,4,5.
        # So k % 3 gives the column within the block.
        
        y_coeffs = [[], [], [], [], [], []] # y00, y01, y02, y10, y11, y12
        for (j, k), c in Y_dict.items():
            r_block = j % 2 # 0 for first row of block, 1 for second
            c_block = k % 3 # 0,1,2
            idx = r_block * 3 + c_block
            y_coeffs[idx].append(((j, k), c))
            
        # Now we have 6 sub-terms for this step.
        # Term 1: x0 * y00 -> z0
        # Term 2: x1 * y10 -> z0
        # Term 3: x0 * y01 -> z1
        # Term 4: x1 * y11 -> z1
        # Term 5: x0 * y02 -> z2
        # Term 6: x1 * y12 -> z2
        
        sub_terms = [
            (x0_coeffs, y_coeffs[0], 0), # x0 * y00 -> z0
            (x1_coeffs, y_coeffs[3], 0), # x1 * y10 -> z0
            (x0_coeffs, y_coeffs[1], 1), # x0 * y01 -> z1
            (x1_coeffs, y_coeffs[4], 1), # x1 * y11 -> z1
            (x0_coeffs, y_coeffs[2], 2), # x0 * y02 -> z2
            (x1_coeffs, y_coeffs[5], 2), # x1 * y12 -> z2
        ]
        
        for X_c, Y_c, z_idx in sub_terms:
            # Fill U
            for (i, j), c in X_c.items():
                U[i * m + j, col] += c
            # Fill V
            for (j, k), c in Y_c:
                V[j * p + k, col] += c
            # Fill W
            # z_idx is 0, 1, or 2.
            # C_targets is list of (block_idx, sign).
            # block_idx 0=C11, 1=C12, 2=C21, 3=C22.
            # C11 is row 0, cols 0,1,2.
            # C12 is row 0, cols 3,4,5.
            # C21 is row 1, cols 0,1,2.
            # C22 is row 1, cols 3,4,5.
            
            for block_idx, sign in C_targets:
                if block_idx == 0: # C11
                    i_C = 0
                    k_C = z_idx
                elif block_idx == 1: # C12
                    i_C = 0
                    k_C = 3 + z_idx
                elif block_idx == 2: # C21
                    i_C = 1
                    k_C = z_idx
                elif block_idx == 3: # C22
                    i_C = 1
                    k_C = 3 + z_idx
                
                W[k_C * n + i_C, col] += sign
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
