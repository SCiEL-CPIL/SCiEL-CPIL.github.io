import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the 4 blocks
    # Block 0: X = A_L + A_R, Y = B_11 + B_21, Target C_L, Factor 1/2
    # Block 1: X = A_L - A_R, Y = B_11 - B_21, Target C_L, Factor 1/2
    # Block 2: X = A_L + A_R, Y = B_12 + B_22, Target C_R, Factor 1/2
    # Block 3: X = A_L - A_R, Y = B_12 - B_22, Target C_R, Factor 1/2
    
    blocks = [
        {
            'x_signs': [(0,0,1), (0,1,1), (1,0,1), (1,1,1)], # A_L + A_R
            'y_signs': [(0,0,1), (0,1,1), (2,0,1), (2,1,1)], # B_11 + B_21
            'target': 'L',
            'factor': 0.5
        },
        {
            'x_signs': [(0,0,1), (0,1,1), (1,0,-1), (1,1,-1)], # A_L - A_R
            'y_signs': [(0,0,1), (0,1,1), (2,0,-1), (2,1,-1)], # B_11 - B_21
            'target': 'L',
            'factor': 0.5
        },
        {
            'x_signs': [(0,0,1), (0,1,1), (1,0,1), (1,1,1)], # A_L + A_R
            'y_signs': [(0,2,1), (0,3,1), (2,2,1), (2,3,1)], # B_12 + B_22
            'target': 'R',
            'factor': 0.5
        },
        {
            'x_signs': [(0,0,1), (0,1,1), (1,0,-1), (1,1,-1)], # A_L - A_R
            'y_signs': [(0,2,1), (0,3,1), (2,2,-1), (2,3,-1)], # B_12 - B_22
            'target': 'R',
            'factor': 0.5
        }
    ]
    
    for b in blocks:
        x_signs = b['x_signs']
        y_signs = b['y_signs']
        target = b['target']
        factor = b['factor']
        
        # Map X indices (0..3) to A indices (0..7)
        # X is 2x2.
        # x[0] -> A[0,0] (idx 0), x[1] -> A[0,1] (idx 1)
        # x[2] -> A[1,0] (idx 2), x[3] -> A[1,1] (idx 3)
        # Wait, A_L is A[:, 0:2].
        # A[0,0] is idx 0. A[0,1] is idx 1.
        # A[1,0] is idx 2. A[1,1] is idx 3.
        # A_R is A[:, 2:4].
        # A[0,2] is idx 4. A[0,3] is idx 5.
        # A[1,2] is idx 6. A[1,3] is idx 7.
        
        # x_signs are given as (i, j, sign) for A[i, j]?
        # My x_signs above were (row, col_in_block, sign).
        # Let's refine x_signs to (a_idx, sign).
        # For A_L + A_R:
        # A_L: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3.
        # A_R: A[0,2]=4, A[0,3]=5, A[1,2]=6, A[1,3]=7.
        # X[0,0] = A[0,0] + A[0,2] -> U[0]+=1, U[4]+=1
        # X[0,1] = A[0,1] + A[0,3] -> U[1]+=1, U[5]+=1
        # X[1,0] = A[1,0] + A[1,2] -> U[2]+=1, U[6]+=1
        # X[1,1] = A[1,1] + A[1,3] -> U[3]+=1, U[7]+=1
        
        # For A_L - A_R:
        # X[0,0] = A[0,0] - A[0,2] -> U[0]+=1, U[4]-=1
        # ...
        
        # Let's define x_coeffs and y_coeffs for the 4 elements of X and Y.
        # x_coeffs[r][c] = list of (a_idx, val)
        # y_coeffs[r][c] = list of (b_idx, val)
        
        x_coeffs = [[[] for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                # X[r,c] is formed from A_L[r,c] and A_R[r,c]
                # A_L[r,c] index: r*2 + c
                # A_R[r,c] index: r*2 + c + 4
                idx_L = r * 2 + c
                idx_R = r * 2 + c + 4
                # Determine signs from block definition
                # Block 0,2: +1, +1
                # Block 1,3: +1, -1
                sign_L = 1
                sign_R = 1 if b in [blocks[0], blocks[2]] else -1
                x_coeffs[r][c].append((idx_L, sign_L))
                x_coeffs[r][c].append((idx_R, sign_R))
                
        y_coeffs = [[[] for _ in range(2)] for _ in range(2)]
        # Y is 2x2.
        # B_11: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
        # B_12: B[0,2]=4, B[0,3]=5, B[1,2]=6, B[1,3]=7
        # B_21: B[2,0]=8, B[2,1]=9, B[3,0]=10, B[3,1]=11
        # B_22: B[2,2]=12, B[2,3]=13, B[3,2]=14, B[3,3]=15
        
        # Block 0: Y = B_11 + B_21
        # Block 1: Y = B_11 - B_21
        # Block 2: Y = B_12 + B_22
        # Block 3: Y = B_12 - B_22
        
        if b == blocks[0]:
            # Y[0,0] = B[0,0] + B[2,0]
            y_coeffs[0][0] = [(0, 1), (8, 1)]
            y_coeffs[0][1] = [(1, 1), (9, 1)]
            y_coeffs[1][0] = [(2, 1), (10, 1)]
            y_coeffs[1][1] = [(3, 1), (11, 1)]
        elif b == blocks[1]:
            y_coeffs[0][0] = [(0, 1), (8, -1)]
            y_coeffs[0][1] = [(1, 1), (9, -1)]
            y_coeffs[1][0] = [(2, 1), (10, -1)]
            y_coeffs[1][1] = [(3, 1), (11, -1)]
        elif b == blocks[2]:
            y_coeffs[0][0] = [(4, 1), (12, 1)]
            y_coeffs[0][1] = [(5, 1), (13, 1)]
            y_coeffs[1][0] = [(6, 1), (14, 1)]
            y_coeffs[1][1] = [(7, 1), (15, 1)]
        elif b == blocks[3]:
            y_coeffs[0][0] = [(4, 1), (12, -1)]
            y_coeffs[0][1] = [(5, 1), (13, -1)]
            y_coeffs[1][0] = [(6, 1), (14, -1)]
            y_coeffs[1][1] = [(7, 1), (15, -1)]
            
        # Strassen Steps
        # M1 = (x00+x11)(y00+y11)
        # M2 = (x10+x11)y00
        # M3 = x00(y10-y11)
        # M4 = x11(y01-y00)
        # M5 = (x00+x01)y11
        # M6 = (x10-x00)(y00+y01)
        # M7 = (x01-x11)(y10+y11)
        
        # Helper to add a product
        def add_product(x_terms, y_terms, w_targets):
            nonlocal col
            # x_terms: list of (a_idx, val)
            # y_terms: list of (b_idx, val)
            # w_targets: list of (c_idx, val)
            
            for a_idx, a_val in x_terms:
                U[a_idx][col] += a_val
            for b_idx, b_val in y_terms:
                V[b_idx][col] += b_val
            for c_idx, c_val in w_targets:
                W[c_idx][col] += c_val * factor # Apply factor here? 
                # Wait, factor is 0.5 for the whole block.
                # It's better to apply factor to W entries.
            col += 1

        # We need to construct x_terms and y_terms for each M.
        
        # M1: (x00+x11)(y00+y11)
        x_terms = x_coeffs[0][0] + x_coeffs[1][1]
        y_terms = y_coeffs[0][0] + y_coeffs[1][1]
        # W targets for M1: z00 += 1, z11 += 1
        # z is 2x2.
        # If target is L, z indices are 0,1,2,3 (C[0,0], C[0,1], C[1,0], C[1,1])
        # If target is R, z indices are 4,5,6,7 (C[0,2], C[0,3], C[1,2], C[1,3])
        if target == 'L':
            z_base = 0
        else:
            z_base = 4
            
        w_targets = [(z_base + 0, 1), (z_base + 3, 1)] # z00, z11
        add_product(x_terms, y_terms, w_targets)
        
        # M2: (x10+x11)y00
        x_terms = x_coeffs[1][0] + x_coeffs[1][1]
        y_terms = y_coeffs[0][0]
        w_targets = [(z_base + 2, 1), (z_base + 3, -1)] # z10, z11
        add_product(x_terms, y_terms, w_targets)
        
        # M3: x00(y10-y11)
        x_terms = x_coeffs[0][0]
        y_terms = y_coeffs[1][0] + [ (idx, -val) for idx, val in y_coeffs[1][1] ]
        w_targets = [(z_base + 1, 1), (z_base + 3, 1)] # z01, z11
        add_product(x_terms, y_terms, w_targets)
        
        # M4: x11(y01-y00)
        x_terms = x_coeffs[1][1]
        y_terms = y_coeffs[0][1] + [ (idx, -val) for idx, val in y_coeffs[0][0] ]
        w_targets = [(z_base + 0, 1)] # z00
        add_product(x_terms, y_terms, w_targets)
        
        # M5: (x00+x01)y11
        x_terms = x_coeffs[0][0] + x_coeffs[0][1]
        y_terms = y_coeffs[1][1]
        w_targets = [(z_base + 0, -1), (z_base + 1, 1)] # z00, z01
        add_product(x_terms, y_terms, w_targets)
        
        # M6: (x10-x00)(y00+y01)
        x_terms = x_coeffs[1][0] + [ (idx, -val) for idx, val in x_coeffs[0][0] ]
        y_terms = y_coeffs[0][0] + y_coeffs[0][1]
        w_targets = [(z_base + 2, 1), (z_base + 3, 1)] # z10, z11
        add_product(x_terms, y_terms, w_targets)
        
        # M7: (x01-x11)(y10+y11)
        x_terms = x_coeffs[0][1] + [ (idx, -val) for idx, val in x_coeffs[1][1] ]
        y_terms = y_coeffs[1][0] + y_coeffs[1][1]
        w_targets = [(z_base + 0, 1)] # z00
        add_product(x_terms, y_terms, w_targets)
        
    return {"U": U, "V": V, "W": W}
