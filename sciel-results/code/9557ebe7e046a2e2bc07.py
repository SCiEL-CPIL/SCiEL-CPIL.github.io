import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 part, Rank 16 for 4x4x1 part
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)] # 16 x 65
    V = [[0] * rank for _ in range(m * p)] # 20 x 65
    W = [[0] * rank for _ in range(p * n)] # 20 x 65
    
    # Strassen coefficients for 2x2 multiplication
    # u[r][a,b], v[r][c,d], w[r][i,j]
    # a,b,c,d,i,j in 0,1
    u_coeffs = [
        [1, 0, 0, 1], # r=1
        [0, 0, 1, 1], # r=2
        [1, 0, 0, 0], # r=3
        [0, 0, 0, 1], # r=4
        [1, 1, 0, 0], # r=5
        [-1, 0, 1, 0], # r=6
        [0, 1, 0, -1]  # r=7
    ]
    v_coeffs = [
        [1, 0, 0, 1], # r=1
        [1, 0, 0, 0], # r=2
        [0, 1, 0, -1], # r=3
        [-1, 0, 1, 0], # r=4
        [0, 0, 0, 1], # r=5
        [1, 1, 0, 0], # r=6
        [0, 0, 1, 1]  # r=7
    ]
    w_coeffs = [
        [1, 0, 0, 1], # r=1
        [0, 0, 1, 0], # r=2
        [0, 1, 0, -1], # r=3
        [-1, 0, 1, 0], # r=4
        [0, 1, 0, 1], # r=5
        [1, 0, 0, -1], # r=6
        [-1, 0, 0, 0]  # r=7
    ]
    
    # Strassen formulas for 4x4 level
    # M_k = X_k Y_k
    # X_k is linear combo of A blocks
    # Y_k is linear combo of B blocks
    # C blocks are linear combos of M_k
    
    # Blocks of A:
    # A00: (0,0)-(1,1)
    # A01: (0,2)-(1,3)
    # A10: (2,0)-(3,1)
    # A11: (2,2)-(3,3)
    
    # We define X_k and Y_k as 2x2 matrices of linear forms
    # Each entry is a list of (coeff, A_index) or (coeff, B_index)
    
    def get_A_idx(r, c):
        return r * m + c
        
    def get_B_idx(r, c):
        return r * p + c
        
    def get_W_idx(r, c):
        # W index is k*n + i ? 
        # Task says W[k*n+i, r]. 
        # Here output is C (p x n). 
        # The task description says "output factor follows the official trace convention W[k*n+i,r]".
        # Usually this means W is indexed by output element.
        # In the seed: W[k*n+i][column] = 1.
        # k is 0..p-1, i is 0..n-1.
        # So index is k*n + i.
        return c * n + r # c is k (col of B, row of C?), r is i (row of A, col of C?)
        # Wait. C is p x n.
        # C[k, i] = sum_j A[i, j] B[j, k].
        # So C index is (k, i).
        # W index should be k*n + i.
        
    # Let's stick to the seed's indexing:
    # U[i*m + j, col]
    # V[j*p + k, col]
    # W[k*n + i, col]
    
    # For the 4x4x4 part:
    # A is 4x4. B is 4x4 (first 4 cols). C is 4x4 (first 4 cols).
    
    # Define the 7 products M_1..M_7
    # Each M_k is a 2x2 matrix.
    # M_k[i, j] is a linear combination of C's blocks.
    
    # We will iterate k=0..6 (for M_1..M_7)
    # and r=0..6 (for the 7 scalar products within M_k)
    
    col = 0
    for k in range(7):
        # Determine X_k and Y_k
        # X_k is 2x2, Y_k is 2x2
        # Each entry is a linear combination of A or B elements
        
        # We'll represent X_k and Y_k as 2x2 arrays of lists of (coeff, index)
        # Or just compute the final linear combination for each scalar product
        
        for r in range(7):
            # Get u, v, w coeffs for this scalar product
            u = u_coeffs[r]
            v = v_coeffs[r]
            w = w_coeffs[r]
            
            # u, v, w are length 4: [00, 01, 10, 11]
            
            # We need to form:
            # sum_{a,b} u[a,b] * X_k[a,b]
            # sum_{c,d} v[c,d] * Y_k[c,d]
            
            # X_k and Y_k are defined by the Strassen formulas at the 4x4 level.
            
            # Let's define the 4x4 Strassen formulas explicitly for X_k and Y_k entries.
            
            # M_1: X = A00 + A11, Y = B00 + B11
            # M_2: X = A10 + A11, Y = B00
            # M_3: X = A00, Y = B01 - B11
            # M_4: X = A11, Y = B10 - B00
            # M_5: X = A00 + A01, Y = B11
            # M_6: X = A10 - A00, Y = B00 + B01
            # M_7: X = A01 - A11, Y = B10 + B11
            
            # We need the linear combination of A elements for the first factor.
            # First factor = sum_{a,b} u[a,b] * X_k[a,b]
            
            # X_k[a,b] is a sum of A elements.
            
            # Let's precompute the linear forms for X_k and Y_k.
            
            # A blocks:
            # A00: [[A00, A01], [A10, A11]] (indices 0,1,4,5)
            # A01: [[A02, A03], [A12, A13]] (indices 2,3,6,7)
            # A10: [[A20, A21], [A30, A31]] (indices 8,9,12,13)
            # A11: [[A22, A23], [A32, A33]] (indices 10,11,14,15)
            
            # B blocks:
            # B00: [[B00, B01], [B10, B11]] (indices 0,1,4,5)
            # B01: [[B02, B03], [B12, B13]] (indices 2,3,6,7)
            # B10: [[B20, B21], [B30, B31]] (indices 8,9,12,13)
            # B11: [[B22, B23], [B32, B33]] (indices 10,11,14,15)
            
            # We'll create a helper to get the linear form.
            
            def get_A_block(block):
                # block: 0=A00, 1=A01, 2=A10, 3=A11
                # returns 2x2 list of (coeff, idx)
                res = [[(0,0), (0,0)], [(0,0), (0,0)]]
                # Map block to base index
                # A00: 0, A01: 2, A10: 8, A11: 10
                base = [0, 2, 8, 10][block]
                # Indices:
                # 00: base
                # 01: base+1
                # 10: base+4
                # 11: base+5
                res[0][0] = (1, base)
                res[0][1] = (1, base+1)
                res[1][0] = (1, base+4)
                res[1][1] = (1, base+5)
                return res

            def get_B_block(block):
                res = [[(0,0), (0,0)], [(0,0), (0,0)]]
                base = [0, 2, 8, 10][block]
                res[0][0] = (1, base)
                res[0][1] = (1, base+1)
                res[1][0] = (1, base+4)
                res[1][1] = (1, base+5)
                return res

            # Define X_k and Y_k as 2x2 arrays of linear forms
            # Linear form is a list of (coeff, index)
            
            X_k = [[[] for _ in range(2)] for _ in range(2)]
            Y_k = [[[] for _ in range(2)] for _ in range(2)]
            
            if k == 0: # M_1
                A00 = get_A_block(0)
                A11 = get_A_block(3)
                B00 = get_B_block(0)
                B11 = get_B_block(3)
                for i in range(2):
                    for j in range(2):
                        X_k[i][j] = A00[i][j] + A11[i][j]
                        Y_k[i][j] = B00[i][j] + B11[i][j]
            elif k == 1: # M_2
                A10 = get_A_block(2)
                A11 = get_A_block(3)
                B00 = get_B_block(0)
                for i in range(2):
                    for j in range(2):
                        X_k[i][j] = A10[i][j] + A11[i][j]
                        Y_k[i][j] = B00[i][j]
            elif k == 2: # M_3
                A00 = get_A_block(0)
                B01 = get_B_block(1)
                B11 = get_B_block(3)
                for i in range(2):
                    for j in range(2):
                        X_k[i][j] = A00[i][j]
                        Y_k[i][j] = B01[i][j]
                        # Subtract B11
                        for coeff, idx in B11[i][j]:
                            Y_k[i][j].append((-coeff, idx))
            elif k == 3: # M_4
                A11 = get_A_block(3)
                B10 = get_B_block(2)
                B00 = get_B_block(0)
                for i in range(2):
                    for j in range(2):
                        X_k[i][j] = A11[i][j]
                        Y_k[i][j] = B10[i][j]
                        for coeff, idx in B00[i][j]:
                            Y_k[i][j].append((-coeff, idx))
            elif k == 4: # M_5
                A00 = get_A_block(0)
                A01 = get_A_block(1)
                B11 = get_B_block(3)
                for i in range(2):
                    for j in range(2):
                        X_k[i][j] = A00[i][j] + A01[i][j]
                        Y_k[i][j] = B11[i][j]
            elif k == 5: # M_6
                A10 = get_A_block(2)
                A00 = get_A_block(0)
                B00 = get_B_block(0)
                B01 = get_B_block(1)
                for i in range(2):
                    for j in range(2):
                        X_k[i][j] = A10[i][j]
                        for coeff, idx in A00[i][j]:
                            X_k[i][j].append((-coeff, idx))
                        Y_k[i][j] = B00[i][j] + B01[i][j]
            elif k == 6: # M_7
                A01 = get_A_block(1)
                A11 = get_A_block(3)
                B10 = get_B_block(2)
                B11 = get_B_block(3)
                for i in range(2):
                    for j in range(2):
                        X_k[i][j] = A01[i][j]
                        for coeff, idx in A11[i][j]:
                            X_k[i][j].append((-coeff, idx))
                        Y_k[i][j] = B10[i][j] + B11[i][j]
            
            # Now compute the linear combination for U
            # sum_{a,b} u[a,b] * X_k[a,b]
            # u is [u00, u01, u10, u11]
            u_lin = {}
            for a in range(2):
                for b in range(2):
                    coeff_u = u[a*2+b]
                    if coeff_u != 0:
                        for coeff_x, idx in X_k[a][b]:
                            total_coeff = coeff_u * coeff_x
                            u_lin[idx] = u_lin.get(idx, 0) + total_coeff
                            
            # Add to U
            for idx, coeff in u_lin.items():
                if coeff != 0:
                    U[idx][col] += coeff
            
            # Compute linear combination for V
            v_lin = {}
            for c in range(2):
                for d in range(2):
                    coeff_v = v[c*2+d]
                    if coeff_v != 0:
                        for coeff_y, idx in Y_k[c][d]:
                            total_coeff = coeff_v * coeff_y
                            v_lin[idx] = v_lin.get(idx, 0) + total_coeff
                            
            for idx, coeff in v_lin.items():
                if coeff != 0:
                    V[idx][col] += coeff
            
            # Compute linear combination for W
            # M_k contributes to C blocks.
            # C_00 = M_1 - M_4 + M_6 - M_7
            # C_01 = M_3 + M_5
            # C_10 = M_2 + M_4
            # C_11 = M_1 - M_3 + M_5 - M_6
            
            # w is [w00, w01, w10, w11] corresponding to M_k's entries
            # M_k[0,0] contributes to C_00[0,0] with coeff 1 (if k=0)
            
            # Let's define the contribution of M_k to C blocks.
            # C_block[0,0] (C00) gets M_k[0,0] * coeff
            # C_block[0,1] (C00's right col? No, C00 is 2x2)
            
            # C is 4x4.
            # C00 is top-left 2x2.
            # C01 is top-right 2x2.
            # C10 is bottom-left 2x2.
            # C11 is bottom-right 2x2.
            
            # M_k is a 2x2 matrix.
            # M_k[0,0] is the top-left entry of the result of the 2x2 multiplication.
            # This entry corresponds to C_XX[0,0].
            
            # Let's map k to the coefficients for C blocks.
            
            # C_00: M_1(1), M_4(-1), M_6(1), M_7(-1)
            # C_01: M_3(1), M_5(1)
            # C_10: M_2(1), M_4(1)
            # C_11: M_1(1), M_3(-1), M_5(1), M_6(-1)
            
            # We need to add w[a,b] * M_k[a,b] to the corresponding C entries.
            
            # M_k[a,b] is a scalar (result of the 2x2 multiplication's (a,b) entry).
            # Wait, M_k is the result of X_k Y_k.
            # The 7 scalar products P_{k,r} are used to compute the entries of M_k.
            # M_k[a,b] = sum_{r=1}^7 w^{(r)}_{ab} P_{k,r}.
            # So for a fixed r, M_k[a,b] gets contribution w^{(r)}_{ab} * P_{k,r}.
            
            # So the contribution to C is:
            # For each a,b in 0..1:
            #   Let w_ab = w[a*2+b]
            #   If w_ab != 0:
            #      The entry M_k[a,b] contributes w_ab * P_{k,r} to C.
            #      Which C entry?
            #      It depends on which block M_k belongs to.
            
            # If k=0 (M_1):
            #   M_1[0,0] -> C_00[0,0] = C[0][0]
            #   M_1[0,1] -> C_00[0,1] = C[0][1]
            #   M_1[1,0] -> C_00[1,0] = C[1][0]
            #   M_1[1,1] -> C_00[1,1] = C[1][1]
            
            #   M_1[0,0] -> C_11[0,0] = C[2][2] ? No.
            #   C_11 = M_1 - M_3 + M_5 - M_6.
            #   So M_1[0,0] also contributes to C_11[0,0] = C[2][2].
            
            # Let's create a mapping from (k, a, b) to list of (C_index, coeff).
            
            def get_W_contrib(k, a, b):
                # k: 0..6
                # a,b: 0..1
                # Returns list of (W_index, coeff)
                # W_index = c*n + r (c is col of C, r is row of C)
                # C is 4x4.
                
                contribs = []
                
                # C_00 (top-left)
                if k == 0: # M_1
                    contribs.append((0*n + 0, 1)) # C[0][0]
                elif k == 3: # M_4
                    contribs.append((0*n + 0, -1))
                elif k == 5: # M_6
                    contribs.append((0*n + 0, 1))
                elif k == 6: # M_7
                    contribs.append((0*n + 0, -1))
                
                # C_00[0,1]
                if k == 0:
                    contribs.append((1*n + 0, 1)) # C[0][1]
                elif k == 3:
                    contribs.append((1*n + 0, -1))
                elif k == 5:
                    contribs.append((1*n + 0, 1))
                elif k == 6:
                    contribs.append((1*n + 0, -1))
                
                # C_00[1,0]
                if k == 0:
                    contribs.append((0*n + 1, 1)) # C[1][0]
                elif k == 3:
                    contribs.append((0*n + 1, -1))
                elif k == 5:
                    contribs.append((0*n + 1, 1))
                elif k == 6:
                    contribs.append((0*n + 1, -1))
                
                # C_00[1,1]
                if k == 0:
                    contribs.append((1*n + 1, 1)) # C[1][1]
                elif k == 3:
                    contribs.append((1*n + 1, -1))
                elif k == 5:
                    contribs.append((1*n + 1, 1))
                elif k == 6:
                    contribs.append((1*n + 1, -1))
                
                # C_01 (top-right)
                if k == 2: # M_3
                    contribs.append((2*n + 0, 1)) # C[0][2]
                elif k == 4: # M_5
                    contribs.append((2*n + 0, 1))
                
                if k == 2:
                    contribs.append((3*n + 0, 1)) # C[0][3]
                elif k == 4:
                    contribs.append((3*n + 0, 1))
                
                if k == 2:
                    contribs.append((2*n + 1, 1)) # C[1][2]
                elif k == 4:
                    contribs.append((2*n + 1, 1))
                
                if k == 2:
                    contribs.append((3*n + 1, 1)) # C[1][3]
                elif k == 4:
                    contribs.append((3*n + 1, 1))
                
                # C_10 (bottom-left)
                if k == 1: # M_2
                    contribs.append((0*n + 2, 1)) # C[2][0]
                elif k == 3: # M_4
                    contribs.append((0*n + 2, 1))
                
                if k == 1:
                    contribs.append((1*n + 2, 1)) # C[2][1]
                elif k == 3:
                    contribs.append((1*n + 2, 1))
                
                if k == 1:
                    contribs.append((0*n + 3, 1)) # C[3][0]
                elif k == 3:
                    contribs.append((0*n + 3, 1))
                
                if k == 1:
                    contribs.append((1*n + 3, 1)) # C[3][1]
                elif k == 3:
                    contribs.append((1*n + 3, 1))
                
                # C_11 (bottom-right)
                if k == 0: # M_1
                    contribs.append((2*n + 2, 1)) # C[2][2]
                elif k == 2: # M_3
                    contribs.append((2*n + 2, -1))
                elif k == 4: # M_5
                    contribs.append((2*n + 2, 1))
                elif k == 5: # M_6
                    contribs.append((2*n + 2, -1))
                
                if k == 0:
                    contribs.append((3*n + 2, 1)) # C[2][3]
                elif k == 2:
                    contribs.append((3*n + 2, -1))
                elif k == 4:
                    contribs.append((3*n + 2, 1))
                elif k == 5:
                    contribs.append((3*n + 2, -1))
                
                if k == 0:
                    contribs.append((2*n + 3, 1)) # C[3][2]
                elif k == 2:
                    contribs.append((2*n + 3, -1))
                elif k == 4:
                    contribs.append((2*n + 3, 1))
                elif k == 5:
                    contribs.append((2*n + 3, -1))
                
                if k == 0:
                    contribs.append((3*n + 3, 1)) # C[3][3]
                elif k == 2:
                    contribs.append((3*n + 3, -1))
                elif k == 4:
                    contribs.append((3*n + 3, 1))
                elif k == 5:
                    contribs.append((3*n + 3, -1))
                
                return contribs

            # Now apply w_coeffs
            for a in range(2):
                for b in range(2):
                    w_ab = w[a*2+b]
                    if w_ab != 0:
                        contribs = get_W_contrib(k, a, b)
                        for w_idx, coeff in contribs:
                            W[w_idx][col] += w_ab * coeff
            
            col += 1
    
    # Add the 16 terms for the last column (4x4x1)
    for i in range(n):
        for j in range(m):
            # C[i, 4] += A[i, j] * B[j, 4]
            # U[i*m+j, col] = 1
            # V[j*p+4, col] = 1
            # W[4*n+i, col] = 1
            
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

